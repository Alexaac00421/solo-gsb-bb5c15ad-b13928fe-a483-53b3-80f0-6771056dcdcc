"""SQLite 持久层。

版本模型
--------
每次导入（步道图或封闭记录）生成一个新版本号：

- 新版本完整复制上一版本的节点/路段/封闭记录，再覆盖被更新的那部分；
- ``meta`` 表中的 ``current_version`` 指针在同一事务内切换；
- 查询通过 :func:`get_snapshot` 在开始时读取一次指针，整次请求固定使用该版本。

导入失败时事务整体回滚，旧版本保持不动。

版本恢复（:func:`restore_version`）把指定历史版本的四类数据整体复制为一个
**新的**生效版本：版本号继续递增、历史版本原样保留，绝不回拨指针。恢复与
导入共用同一 ``BEGIN IMMEDIATE`` 写事务，彼此串行，生效顺序即提交顺序。

图发布世代（generation）
------------------------
行程容量按**图发布世代**独立计量：

- 仅更新封闭记录（:func:`publish_closures`）不开新世代，已确认行程的容量
  占用继续计入当前世代；
- 重新发布步道图（:func:`publish_graph`）或恢复历史版本
  （:func:`restore_version`）开启一个递增的新世代，新世代初始没有任何
  占用（旧世代行程仍可查询、取消）；
- 每个行程在确认时绑定当时的世代与数据版本，容量检查只统计**同一世代内
  未取消**行程的逐段半开区间占用。
"""
import json
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Optional

from . import config
from .schemas import ValidationError, normalize_closures, normalize_graph

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK (kind IN ('graph', 'closures', 'restore')),
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS nodes (
    version_id  INTEGER NOT NULL,
    node_id     TEXT NOT NULL,
    PRIMARY KEY (version_id, node_id)
);

CREATE TABLE IF NOT EXISTS edges (
    version_id      INTEGER NOT NULL,
    edge_id         TEXT NOT NULL,
    from_node       TEXT NOT NULL,
    to_node         TEXT NOT NULL,
    travel_seconds  INTEGER NOT NULL CHECK (travel_seconds > 0),
    has_stairs      INTEGER NOT NULL CHECK (has_stairs IN (0, 1)),
    capacity        INTEGER,
    PRIMARY KEY (version_id, edge_id)
);

CREATE TABLE IF NOT EXISTS closures (
    version_id  INTEGER NOT NULL,
    closure_id  INTEGER NOT NULL,
    edge_id     TEXT NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL CHECK (end_ts > start_ts),
    PRIMARY KEY (version_id, closure_id)
);

CREATE TABLE IF NOT EXISTS reverse_windows (
    version_id  INTEGER NOT NULL,
    edge_id     TEXT NOT NULL,
    window_idx  INTEGER NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL CHECK (end_ts > start_ts),
    PRIMARY KEY (version_id, edge_id, window_idx)
);

-- 已确认行程（容量占用由 trip_segments 派生；取消只改状态，可重复调用）
CREATE TABLE IF NOT EXISTS trips (
    trip_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    status         TEXT NOT NULL DEFAULT 'confirmed'
                       CHECK (status IN ('confirmed', 'cancelled')),
    generation_id  INTEGER NOT NULL,
    version_id     INTEGER NOT NULL,
    origin         TEXT NOT NULL,
    destination    TEXT NOT NULL,
    departure_ts   INTEGER NOT NULL,
    arrival_ts     INTEGER NOT NULL,
    avoid_stairs   INTEGER NOT NULL CHECK (avoid_stairs IN (0, 1)),
    party_size     INTEGER NOT NULL CHECK (party_size > 0),
    created_at     INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    cancelled_at   INTEGER
);

CREATE TABLE IF NOT EXISTS trip_segments (
    trip_id    INTEGER NOT NULL,
    seq        INTEGER NOT NULL,
    edge_id    TEXT NOT NULL,
    from_node  TEXT NOT NULL,
    to_node    TEXT NOT NULL,
    enter_ts   INTEGER NOT NULL,
    leave_ts   INTEGER NOT NULL,
    PRIMARY KEY (trip_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_edges_from     ON edges(version_id, from_node);
CREATE INDEX IF NOT EXISTS idx_closures_edge ON closures(version_id, edge_id);
CREATE INDEX IF NOT EXISTS idx_reverse_edge  ON reverse_windows(version_id, edge_id);
CREATE INDEX IF NOT EXISTS idx_trip_seg_edge
    ON trip_segments(edge_id, enter_ts, leave_ts);
CREATE INDEX IF NOT EXISTS idx_trips_gen ON trips(generation_id, status);
"""


class EdgeRow(NamedTuple):
    id: str
    src: str
    dst: str
    seconds: int
    has_stairs: bool
    capacity: Optional[int] = None  # None 表示不限制通行容量


class ClosureRow(NamedTuple):
    edge_id: str
    start: int
    end: int


class ReverseWindowRow(NamedTuple):
    edge_id: str
    start: int
    end: int


class TripSegmentRow(NamedTuple):
    seq: int
    edge_id: str
    from_node: str
    to_node: str
    enter: int
    leave: int


class TripRow(NamedTuple):
    """一条已确认（或已取消）行程的持久化记录。"""

    trip_id: int
    status: str                # 'confirmed' | 'cancelled'
    generation_id: int
    version_id: int
    origin: str
    destination: str
    departure: int
    arrival: int
    avoid_stairs: bool
    party_size: int
    segments: tuple[TripSegmentRow, ...] = ()


class StaleVersionError(ValueError):
    """确认请求携带的数据版本与当前生效版本不一致（不写入任何占用）。"""


class Snapshot(NamedTuple):
    """单次查询使用的不可变数据版本。"""

    version_id: int
    nodes: list[str]
    edges: list[EdgeRow]
    closures: list[ClosureRow]
    reverse_windows: list[ReverseWindowRow] = []

    @property
    def node_set(self) -> set[str]:
        return set(self.nodes)


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表并写入初始指针（不存在任何已发布版本时）。"""
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('current_version', '0')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('current_generation', '0')"
    )
    _migrate_versions_kind(conn)
    _migrate_edges_capacity(conn)


def _migrate_edges_capacity(conn: sqlite3.Connection) -> None:
    """旧库 ``edges`` 表缺少 ``capacity`` 列时原地补列。

    已有行的容量为 NULL（不限制），与“缺省不限”语义一致。
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(edges)")}
    if "capacity" in cols:
        return
    conn.execute("ALTER TABLE edges ADD COLUMN capacity INTEGER")


def _migrate_versions_kind(conn: sqlite3.Connection) -> None:
    """旧库的 ``versions.kind`` CHECK 不含 ``'restore'`` 时，原地重建该表。

    建表语句中的 CHECK 约束无法就地修改，按 SQLite 惯例
    “建新表 → 复制 → 删旧表 → 改名”迁移；已有版本行原样保留。
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='versions'"
    ).fetchone()
    if row is None or "'restore'" in row["sql"]:
        return
    with transaction(conn):
        conn.execute("DROP TABLE IF EXISTS versions_new")
        conn.execute(
            "CREATE TABLE versions_new ("
            "    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    kind        TEXT NOT NULL CHECK (kind IN ('graph', 'closures', 'restore')),"
            "    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))"
            ")"
        )
        conn.execute(
            "INSERT INTO versions_new(version_id, kind, created_at) "
            "SELECT version_id, kind, created_at FROM versions"
        )
        conn.execute("DROP TABLE versions")
        conn.execute("ALTER TABLE versions_new RENAME TO versions")


@contextmanager
def transaction(conn: sqlite3.Connection):
    """立即失败的写事务，避免与其他导入互相等待。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key='current_version'"
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def current_generation(conn: sqlite3.Connection) -> int:
    """当前图发布世代号（尚无图发布时为 0）。"""
    row = conn.execute(
        "SELECT value FROM meta WHERE key='current_generation'"
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def _begin_generation(conn: sqlite3.Connection) -> int:
    """在写事务内开启下一个图发布世代并切换指针，返回新世代号。"""
    new_gen = current_generation(conn) + 1
    conn.execute(
        "UPDATE meta SET value=? WHERE key='current_generation'", (str(new_gen),)
    )
    return new_gen


def _copy_version(conn: sqlite3.Connection, old: int, new: int, kind: str) -> None:
    conn.execute("INSERT INTO versions(version_id, kind) VALUES (?, ?)", (new, kind))
    if old:
        conn.execute("INSERT INTO nodes(version_id, node_id) SELECT ?, node_id FROM nodes WHERE version_id=?", (new, old))
        conn.execute(
            "INSERT INTO edges(version_id, edge_id, from_node, to_node, travel_seconds, has_stairs, capacity) "
            "SELECT ?, edge_id, from_node, to_node, travel_seconds, has_stairs, capacity FROM edges WHERE version_id=?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO closures(version_id, closure_id, edge_id, start_ts, end_ts) "
            "SELECT ?, closure_id, edge_id, start_ts, end_ts FROM closures WHERE version_id=?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO reverse_windows(version_id, edge_id, window_idx, start_ts, end_ts) "
            "SELECT ?, edge_id, window_idx, start_ts, end_ts FROM reverse_windows "
            "WHERE version_id=?",
            (new, old),
        )


def publish_graph(conn: sqlite3.Connection, payload) -> int:
    """原子发布新步道图。

    校验失败抛出 :class:`ValidationError` 并完整回滚；旧封闭记录被保留，
    其中引用了不在新图中路段的记录会自动丢弃。
    """
    graph = normalize_graph(payload)
    new_edge_ids = {e["id"] for e in graph["edges"]}

    with transaction(conn):
        old = _current_version(conn)
        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, old, new, "graph")

        # 用新版本的图整体替换
        conn.execute("DELETE FROM nodes WHERE version_id=?", (new,))
        conn.execute("DELETE FROM edges WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO nodes(version_id, node_id) VALUES (?, ?)",
            [(new, n) for n in graph["nodes"]],
        )
        conn.executemany(
            "INSERT INTO edges(version_id, edge_id, from_node, to_node, travel_seconds, has_stairs, capacity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (new, e["id"], e["src"], e["dst"], e["seconds"], 1 if e["has_stairs"] else 0,
                 e.get("capacity"))
                for e in graph["edges"]
            ],
        )
        # 反向时间窗随新图整体替换（校验已在事务前完成）
        conn.execute("DELETE FROM reverse_windows WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO reverse_windows(version_id, edge_id, window_idx, start_ts, end_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (new, e["id"], idx, start, end)
                for e in graph["edges"]
                for idx, (start, end) in enumerate(e["reverse_windows"])
            ],
        )
        # 丢弃悬挂封闭记录
        conn.execute("DELETE FROM closures WHERE version_id=? AND edge_id NOT IN (%s)" %
                     (",".join("?" * len(new_edge_ids)) if new_edge_ids else "''"),
                     (new, *sorted(new_edge_ids)) if new_edge_ids else (new,))

        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
        # 重新发布图开启独立计量的新图发布世代
        _begin_generation(conn)
    return new


def publish_closures(conn: sqlite3.Connection, payload) -> int:
    """原子整体替换封闭记录。必须先发布过步道图。"""
    with transaction(conn):
        old = _current_version(conn)
        if not old:
            # 回滚由 contextmanager 完成
            raise ValueError("尚未导入步道图，无法发布封闭记录")
        known_edges = {
            r["edge_id"]
            for r in conn.execute("SELECT edge_id FROM edges WHERE version_id=?", (old,))
        }
        closures = normalize_closures(payload, known_edges)

        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, old, new, "closures")
        conn.execute("DELETE FROM closures WHERE version_id=?", (new,))
        conn.executemany(
            "INSERT INTO closures(version_id, closure_id, edge_id, start_ts, end_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [(new, i, c["edge_id"], c["start"], c["end"]) for i, c in enumerate(closures)],
        )
        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
    return new


class VersionNotFoundError(ValueError):
    """恢复目标版本不存在（恢复被拒绝，生效版本不变）。"""


def restore_version(conn: sqlite3.Connection, target: int) -> int:
    """把历史版本 ``target`` 整体恢复为新的生效版本。

    在单个写事务中完成：校验目标版本存在 → 分配递增的新版本号 → 完整复制
    目标版本的节点/路段/封闭记录/反向通行窗 → 切换 ``current_version`` 指针。
    目标版本本身原样保留（历史仍可追溯），指针只前进不回拨。

    目标不存在时抛出 :class:`VersionNotFoundError` 并整体回滚，生效版本不变；
    目标版本号非法（非正整数）时抛出 :class:`ValidationError`。
    """
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValidationError(f"恢复目标版本号必须是正整数，收到 {target!r}")
    with transaction(conn):
        exists = conn.execute(
            "SELECT 1 FROM versions WHERE version_id=?", (target,)
        ).fetchone()
        if exists is None:
            raise VersionNotFoundError(f"版本 v{target} 不存在，无法恢复")
        new = int(
            conn.execute("SELECT COALESCE(MAX(version_id), 0) + 1 AS v FROM versions").fetchone()["v"]
        )
        _copy_version(conn, target, new, "restore")
        conn.execute(
            "UPDATE meta SET value=? WHERE key='current_version'", (str(new),)
        )
        # 恢复历史版本视为重新发布图：新世代独立计量，旧行程仍可查询/取消
        _begin_generation(conn)
    return new


def get_snapshot(conn: sqlite3.Connection, version_id: Optional[int] = None) -> Optional[Snapshot]:
    """读取某个版本（默认当前版本）的完整数据。无已发布版本返回 None。"""
    if version_id is None:
        version_id = _current_version(conn)
    if not version_id:
        return None
    exists = conn.execute(
        "SELECT 1 FROM versions WHERE version_id=?", (version_id,)
    ).fetchone()
    if exists is None:
        return None

    nodes = [r["node_id"] for r in conn.execute(
        "SELECT node_id FROM nodes WHERE version_id=? ORDER BY node_id", (version_id,))]
    edges = [
        EdgeRow(r["edge_id"], r["from_node"], r["to_node"], r["travel_seconds"],
                bool(r["has_stairs"]), r["capacity"])
        for r in conn.execute(
            "SELECT edge_id, from_node, to_node, travel_seconds, has_stairs, capacity "
            "FROM edges WHERE version_id=?", (version_id,))
    ]
    closures = [
        ClosureRow(r["edge_id"], r["start_ts"], r["end_ts"])
        for r in conn.execute(
            "SELECT edge_id, start_ts, end_ts FROM closures WHERE version_id=? "
            "ORDER BY closure_id", (version_id,))
    ]
    reverse_windows = [
        ReverseWindowRow(r["edge_id"], r["start_ts"], r["end_ts"])
        for r in conn.execute(
            "SELECT edge_id, start_ts, end_ts FROM reverse_windows WHERE version_id=? "
            "ORDER BY edge_id, window_idx", (version_id,))
    ]
    return Snapshot(version_id, nodes, edges, closures, reverse_windows)


def current_version(conn: sqlite3.Connection) -> int:
    return _current_version(conn)


# ----------------------------------------------------------------------
# 已确认行程：容量占用、原子确认、查询、取消
# ----------------------------------------------------------------------
def active_occupancy(conn: sqlite3.Connection, generation_id: int) -> dict[str, list[tuple[int, int, int]]]:
    """读取某世代内所有**未取消**行程的逐段占用。

    返回 ``{edge_id: [(enter_ts, leave_ts, party_size), ...]}``，按进入时刻
    排序。占用区间为半开 ``[enter, leave)``。确认行程在写事务内调用：
    由于所有确认都经同一 ``BEGIN IMMEDIATE`` 串行提交，读到的占用集即本次
    规划与写入的一致依据，并发确认不可能超额。
    """
    rows = conn.execute(
        "SELECT s.edge_id AS edge_id, s.enter_ts AS enter_ts, s.leave_ts AS leave_ts, "
        "       t.party_size AS party_size "
        "FROM trip_segments s JOIN trips t ON t.trip_id = s.trip_id "
        "WHERE t.generation_id=? AND t.status='confirmed' "
        "ORDER BY s.enter_ts, s.leave_ts",
        (generation_id,),
    ).fetchall()
    occupancy: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for r in rows:
        occupancy[r["edge_id"]].append((r["enter_ts"], r["leave_ts"], r["party_size"]))
    return dict(occupancy)


def confirm_trip(
    conn: sqlite3.Connection,
    request: dict,
) -> TripRow:
    """原子确认一条容量可行的行程。

    ``request`` 为 :func:`app.schemas.normalize_trip_request` 的规范化结果
    （origin/destination/departure/avoid_stairs/party_size/expected_version）。

    在单个 ``BEGIN IMMEDIATE`` 写事务内：

    1. 若请求携带 ``expected_version``，校验其等于当前生效版本，
       否则抛 :class:`StaleVersionError`（不写入任何内容）；
    2. 读取当前图发布世代内未取消行程的占用，用容量约束规划；
    3. 规划成功才写入行程与逐段记录——规划与写入同一事务，确认彼此
       串行，故不可能超额；规划失败（``NoRouteError``）整体回滚，不留占用。
    """
    from .routing import build_engine  # 避免循环导入

    with transaction(conn):
        version = _current_version(conn)
        if not version:
            raise ValueError("尚未导入步道图，无法确认行程")
        expected_version = request.get("expected_version")
        if expected_version is not None and expected_version != version:
            raise StaleVersionError(
                f"数据版本已变化：请求基于 v{expected_version}，当前生效 v{version}，"
                "请基于新版本重新规划后再确认"
            )
        snap = get_snapshot(conn, version)
        generation_id = current_generation(conn)
        occupancy = active_occupancy(conn, generation_id)
        engine = build_engine(snap, occupancy=occupancy)
        party_size = request["party_size"]
        planned = engine.plan(
            origin=request["origin"],
            destination=request["destination"],
            departure=request["departure"],
            avoid_stairs=request["avoid_stairs"],
            party_size=party_size,
        )

        cur = conn.execute(
            "INSERT INTO trips(generation_id, version_id, origin, destination, "
            "departure_ts, arrival_ts, avoid_stairs, party_size, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')",
            (generation_id, version, planned.origin, planned.destination,
             planned.departure_time, planned.arrival_time,
             1 if planned.avoid_stairs else 0, party_size),
        )
        trip_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO trip_segments(trip_id, seq, edge_id, from_node, to_node, "
            "enter_ts, leave_ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (trip_id, i, seg.edge_id, seg.from_node, seg.to_node,
                 seg.enter_time, seg.leave_time)
                for i, seg in enumerate(planned.segments)
            ],
        )
    return TripRow(
        trip_id=trip_id,
        status="confirmed",
        generation_id=generation_id,
        version_id=version,
        origin=planned.origin,
        destination=planned.destination,
        departure=planned.departure_time,
        arrival=planned.arrival_time,
        avoid_stairs=planned.avoid_stairs,
        party_size=party_size,
        segments=tuple(
            TripSegmentRow(i, seg.edge_id, seg.from_node, seg.to_node,
                           seg.enter_time, seg.leave_time)
            for i, seg in enumerate(planned.segments)
        ),
    )


def get_trip(conn: sqlite3.Connection, trip_id: int) -> Optional[TripRow]:
    """按编号读取行程（含已取消行程），不存在返回 None。"""
    row = conn.execute(
        "SELECT * FROM trips WHERE trip_id=?", (trip_id,)
    ).fetchone()
    if row is None:
        return None
    seg_rows = conn.execute(
        "SELECT seq, edge_id, from_node, to_node, enter_ts, leave_ts "
        "FROM trip_segments WHERE trip_id=? ORDER BY seq", (trip_id,)
    ).fetchall()
    return TripRow(
        trip_id=row["trip_id"],
        status=row["status"],
        generation_id=row["generation_id"],
        version_id=row["version_id"],
        origin=row["origin"],
        destination=row["destination"],
        departure=row["departure_ts"],
        arrival=row["arrival_ts"],
        avoid_stairs=bool(row["avoid_stairs"]),
        party_size=row["party_size"],
        segments=tuple(
            TripSegmentRow(r["seq"], r["edge_id"], r["from_node"], r["to_node"],
                           r["enter_ts"], r["leave_ts"])
            for r in seg_rows
        ),
    )


def cancel_trip(conn: sqlite3.Connection, trip_id: int) -> Optional[TripRow]:
    """取消行程（幂等）。

    已确认 → 在写事务内置为 cancelled（容量占用随之不再计入）；
    已取消 → 直接返回当前记录，不重复释放；不存在 → 返回 None。
    """
    with transaction(conn):
        row = conn.execute(
            "SELECT status FROM trips WHERE trip_id=?", (trip_id,)
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "confirmed":
            conn.execute(
                "UPDATE trips SET status='cancelled', "
                "cancelled_at=strftime('%s','now') WHERE trip_id=?",
                (trip_id,),
            )
    return get_trip(conn, trip_id)


def export_json(conn: sqlite3.Connection, path: Path) -> None:
    """辅助：把当前版本导出为 JSON（CLI 排错用）。"""
    snap = get_snapshot(conn)
    if snap is None:
        raise RuntimeError("当前没有已发布的数据版本")
    path.write_text(
        json.dumps(
            {
                "version_id": snap.version_id,
                "nodes": snap.nodes,
                "edges": [e._asdict() | {"has_stairs": e.has_stairs} for e in snap.edges],
                "closures": [c._asdict() for c in snap.closures],
                "reverse_windows": [w._asdict() for w in snap.reverse_windows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
