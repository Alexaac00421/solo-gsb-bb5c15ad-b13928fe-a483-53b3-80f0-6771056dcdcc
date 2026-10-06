"""高峰时段有容量保障的行程确认。

容量语义
========
- 步道图每条路段可携带正整数通行容量 ``capacity``，缺省/``null`` 不限；
- 已确认行程按每段**实际进入至离开**的半开区间 ``[enter, leave)`` 占用
  相应人数的容量；同一段在同一时刻的占用人数之和不得超过容量；
- 容量不足时允许在节点等待（时间依赖 FIFO 网络），等待规则与封闭、
  反向通行窗一致：按**进入时刻**判定，只推迟进入时刻；
- 容量对两个通行方向同样生效（路段容量不分方向）。

确认与并发
==========
确认在单个 ``BEGIN IMMEDIATE`` 写事务内完成“读取当前版本与当前世代的
全部占用 → 重新规划 → 校验容量 → 原子写入行程”。SQLite 写事务全局
串行，故并发确认看到的占用互不重叠，绝不可能超额；规划不可行时事务
整体回滚，不留下任何占用。

版本与世代
==========
- 客户端可在请求中携带 ``data_version``：与服务端当前版本不一致时
  **不写入**，抛出 :class:`VersionConflictError`，由调用方基于新版本
  重新规划或显式放弃；不传则总是按当前版本规划；
- 容量按**图发布世代**计量：步道图发布（``/admin/graph``）与版本恢复
  （``/admin/restore``）开启新世代；封闭记录更新沿用当前世代，其中
  已确认行程继续参与计量。重新发布/恢复后的新世代独立计量，旧行程
  仍可查询与取消（取消只释放其所属世代的容量）。
"""
from __future__ import annotations

import sqlite3
import time
from typing import Optional

from . import db
from .db import Snapshot, TripRecord, transaction
from .routing import NoRouteError, Route, build_engine


class VersionConflictError(Exception):
    """请求基于的数据版本已变化，未写入任何行程。"""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"数据版本已变化：请求基于 v{expected}，当前为 v{current}，"
            "请基于新版本重新规划后再确认"
        )
        self.expected_version = expected
        self.current_version = current


def _segments_to_rows(route: Route) -> list[tuple]:
    return [
        (
            seg.edge_id,
            seg.from_node,
            seg.to_node,
            seg.enter_time,
            seg.leave_time,
            seg.travel_seconds,
            seg.waiting_seconds,
            seg.has_stairs,
        )
        for seg in route.segments
    ]


def confirm_trip(
    conn: sqlite3.Connection,
    *,
    origin: str,
    destination: str,
    departure: int,
    avoid_stairs: bool,
    people: int,
    expected_version: Optional[int] = None,
) -> tuple[TripRecord, Snapshot]:
    """原子确认一条有容量保障的行程。

    必须在外层**写事务**中调用（本函数内部不再加事务），以保证
    “读取占用 → 规划 → 写入”与并发确认/取消/发布严格串行。

    成功返回 ``(行程记录, 规划使用的版本快照)``；版本不匹配抛
    :class:`VersionConflictError`；容量/封闭/窗约束下无可行行程抛
    :class:`NoRouteError`——两种失败均不写入任何内容（调用方回滚）。
    """
    # 写事务内读取当前版本与世代：与所有其他写入严格串行
    snap = db.get_snapshot(conn)
    if snap is None:
        raise NoRouteError("尚未导入步道图，无法确认行程")
    if expected_version is not None and expected_version != snap.version_id:
        raise VersionConflictError(expected_version, snap.version_id)

    occupancy = db.active_occupancy(conn, snap.generation_id)
    engine = build_engine(snap, occupancy)
    route = engine.plan(
        origin=origin,
        destination=destination,
        departure=departure,
        avoid_stairs=avoid_stairs,
        party_size=people,
    )

    trip_id = db.insert_trip(
        conn,
        version_id=snap.version_id,
        generation_id=snap.generation_id,
        origin=origin,
        destination=destination,
        departure_ts=departure,
        arrival_ts=route.arrival_time,
        avoid_stairs=avoid_stairs,
        party_size=people,
        segments=_segments_to_rows(route),
    )
    record = db.get_trip(conn, trip_id)
    return record, snap


def confirm_trip_atomic(conn: sqlite3.Connection, **kwargs) -> tuple[TripRecord, Snapshot]:
    """打开立即写事务后执行 :func:`confirm_trip`；失败整体回滚。"""
    with transaction(conn):
        return confirm_trip(conn, **kwargs)


def cancel_trip_atomic(conn: sqlite3.Connection, trip_id: int) -> Optional[tuple[TripRecord, bool]]:
    """取消行程并释放容量（幂等）。

    返回 ``(行程记录, released)``：行程不存在返回 ``None``；首次取消
    ``released=True``（在单个写事务内把行程置为取消、释放其世代容量）；
    重复取消 ``released=False``，不重复释放。
    """
    with transaction(conn):
        record = db.get_trip(conn, trip_id)
        if record is None:
            return None
        released = db.cancel_trip(conn, trip_id, int(time.time()))
        if not released:
            # 此前已取消：原样返回，不做任何写入
            record = db.get_trip(conn, trip_id)
            return record, False
        return db.get_trip(conn, trip_id), True
