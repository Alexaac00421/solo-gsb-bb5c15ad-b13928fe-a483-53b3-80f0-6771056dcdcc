"""已确认行程测试：原子确认、容量不超额、查询、取消、世代与版本语义。"""
import os
import tempfile
import threading
import unittest
from pathlib import Path

from app.db import (
    StaleVersionError,
    active_occupancy,
    cancel_trip,
    confirm_trip,
    connect,
    current_generation,
    current_version,
    get_snapshot,
    get_trip,
    init_db,
    publish_closures,
    publish_graph,
    restore_version,
)
from app.routing import NoRouteError
from app.schemas import ValidationError, normalize_graph, normalize_trip_request

GRAPH = {
    "nodes": ["A", "B", "C", "D"],
    "edges": [
        {"id": "E1", "from": "A", "to": "B", "travel_seconds": 60,
         "has_stairs": False, "capacity": 1},
        {"id": "E2", "from": "B", "to": "C", "travel_seconds": 60,
         "has_stairs": False},
        {"id": "E3", "from": "A", "to": "D", "travel_seconds": 60,
         "has_stairs": True, "capacity": 2},
    ],
}


def req(origin="A", destination="B", departure=1000, avoid=False, size=1,
        version=None):
    payload = {
        "origin": origin, "destination": destination,
        "departure_time": departure, "avoid_stairs": avoid,
        "party_size": size,
    }
    if version is not None:
        payload["data_version"] = version
    return normalize_trip_request(payload, {"A", "B", "C", "D"})


class TripTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db_path)
        init_db(self.conn)
        publish_graph(self.conn, GRAPH)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()
        os.environ.pop("APP_DB_PATH", None)


class ConfirmTests(TripTestBase):
    def test_confirm_returns_persisted_trip(self):
        trip = confirm_trip(self.conn, req())
        self.assertEqual(trip.status, "confirmed")
        self.assertEqual(trip.party_size, 1)
        self.assertEqual(trip.version_id, 1)
        self.assertEqual(trip.generation_id, 1)
        self.assertEqual(len(trip.segments), 1)
        seg = trip.segments[0]
        self.assertEqual((seg.edge_id, seg.enter, seg.leave), ("E1", 1000, 1060))
        fetched = get_trip(self.conn, trip.trip_id)
        self.assertEqual(fetched.arrival, 1060)

    def test_confirm_serializes_capacity(self):
        t1 = confirm_trip(self.conn, req())
        self.assertEqual(t1.segments[0].enter, 1000)
        t2 = confirm_trip(self.conn, req())
        self.assertEqual(t2.segments[0].enter, 1060)
        t3 = confirm_trip(self.conn, req())
        self.assertEqual(t3.segments[0].enter, 1120)

    def test_confirm_capacity_party_size_aggregates(self):
        # 容量 2 的 E3：一批 2 人占满，第二批须等待
        confirm_trip(self.conn, req(origin="A", destination="D", size=2))
        t = confirm_trip(self.conn, req(origin="A", destination="D", size=1))
        self.assertEqual(t.segments[0].enter, 1060)

    def test_confirm_unlimited_capacity_edge(self):
        # E2 缺省不限：连续多批同时进入
        for _ in range(5):
            t = confirm_trip(self.conn, req(origin="B", destination="C"))
            self.assertEqual(t.segments[0].enter, 1000)

    def test_confirm_reject_when_party_exceeds_capacity(self):
        with self.assertRaises(NoRouteError):
            confirm_trip(self.conn, req(size=2))  # E1 容量 1
        # 拒绝不留占用：随后 1 人仍可立即进入
        t = confirm_trip(self.conn, req())
        self.assertEqual(t.segments[0].enter, 1000)

    def test_avoid_stairs_route_uses_capacity(self):
        # A->C 必须走 E1->E2；E1 容量 1 被占用后避台阶队伍排队
        confirm_trip(self.conn, req())
        t = confirm_trip(self.conn, req(origin="A", destination="C"))
        self.assertEqual([s.edge_id for s in t.segments], ["E1", "E2"])
        self.assertEqual(t.segments[0].enter, 1060)
        self.assertEqual(t.segments[1].enter, 1120)
        self.assertEqual(t.arrival, 1180)

    def test_concurrent_confirms_never_overbook(self):
        n = 8
        results = []
        errors = []

        def worker():
            conn = connect(self.db_path)
            try:
                trip = confirm_trip(conn, req())
                results.append((trip.trip_id, trip.segments[0].enter,
                                trip.segments[0].leave))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                conn.close()

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), n)
        results.sort(key=lambda r: r[1])
        enters = [r[1] for r in results]
        self.assertEqual(enters, [1000 + 60 * i for i in range(n)])
        # 任意两段半开区间不得重叠（容量 1）
        for (_, s1, e1), (_, s2, e2) in zip(results, results[1:]):
            self.assertLessEqual(e1, s2)
        occ = active_occupancy(self.conn, current_generation(self.conn))
        self.assertEqual(len(occ["E1"]), n)


class CancelTests(TripTestBase):
    def test_get_unknown_trip_is_none(self):
        self.assertIsNone(get_trip(self.conn, 999))
        self.assertIsNone(cancel_trip(self.conn, 999))

    def test_cancel_is_idempotent(self):
        trip = confirm_trip(self.conn, req())
        cancelled = cancel_trip(self.conn, trip.trip_id)
        self.assertEqual(cancelled.status, "cancelled")
        again = cancel_trip(self.conn, trip.trip_id)
        self.assertEqual(again.status, "cancelled")
        # 取消行程不再计入占用
        occ = active_occupancy(self.conn, trip.generation_id)
        self.assertNotIn("E1", occ)

    def test_cancel_releases_capacity_once(self):
        t1 = confirm_trip(self.conn, req())           # [1000,1060)
        t2 = confirm_trip(self.conn, req())           # [1060,1120)
        # 未取消时新队伍只能排到 1120
        t3 = confirm_trip(self.conn, req())
        self.assertEqual(t3.segments[0].enter, 1120)
        # 取消 t1：[1000,1060) 槽位释放；重复取消不重复释放
        cancel_trip(self.conn, t1.trip_id)
        cancel_trip(self.conn, t1.trip_id)
        occ = active_occupancy(self.conn, t1.generation_id)
        self.assertEqual(occ["E1"], [(1060, 1120, 1), (1120, 1180, 1)])
        # 新队伍可使用刚释放的 [1000,1060) 槽位（与 t2 半开相接不冲突）
        t4 = confirm_trip(self.conn, req())
        self.assertEqual(t4.segments[0].enter, 1000)

    def test_cancelled_trip_still_queryable(self):
        trip = confirm_trip(self.conn, req())
        cancel_trip(self.conn, trip.trip_id)
        fetched = get_trip(self.conn, trip.trip_id)
        self.assertEqual(fetched.status, "cancelled")
        self.assertEqual(fetched.arrival, 1060)


class GenerationTests(TripTestBase):
    def test_closures_update_keeps_generation(self):
        gen0 = current_generation(self.conn)
        confirm_trip(self.conn, req())
        v = publish_closures(self.conn, {"closures": []})
        self.assertEqual(current_generation(self.conn), gen0)
        # 同世代：既有容量占用继续计量 → 新队伍排到 1060
        t = confirm_trip(self.conn, req())
        self.assertEqual(t.version_id, v)
        self.assertEqual(t.generation_id, gen0)
        self.assertEqual(t.segments[0].enter, 1060)

    def test_graph_republish_starts_independent_generation(self):
        old = confirm_trip(self.conn, req())
        gen_old = current_generation(self.conn)
        publish_graph(self.conn, GRAPH)  # 内容相同也开新世代
        gen_new = current_generation(self.conn)
        self.assertEqual(gen_new, gen_old + 1)
        # 新世代无占用：立即进入
        t = confirm_trip(self.conn, req())
        self.assertEqual(t.generation_id, gen_new)
        self.assertEqual(t.segments[0].enter, 1000)
        # 旧行程仍可查询、取消
        self.assertEqual(get_trip(self.conn, old.trip_id).status, "confirmed")
        cancel_trip(self.conn, old.trip_id)
        # 取消旧世代行程不影响新世代占用
        t2 = confirm_trip(self.conn, req())
        self.assertEqual(t2.generation_id, gen_new)
        self.assertEqual(t2.segments[0].enter, 1060)

    def test_restore_starts_new_generation(self):
        old = confirm_trip(self.conn, req())
        gen_old = current_generation(self.conn)
        publish_closures(self.conn, {"closures": []})
        new_version = restore_version(self.conn, 1)
        self.assertEqual(current_generation(self.conn), gen_old + 1)
        # 新世代独立计量
        t = confirm_trip(self.conn, req())
        self.assertEqual(t.version_id, new_version)
        self.assertEqual(t.segments[0].enter, 1000)
        # 旧行程仍可查询/取消
        self.assertEqual(get_trip(self.conn, old.trip_id).status, "confirmed")
        self.assertEqual(cancel_trip(self.conn, old.trip_id).status, "cancelled")


class VersionConflictTests(TripTestBase):
    def test_stale_expected_version_rejected_without_write(self):
        publish_closures(self.conn, {"closures": []})  # v2
        with self.assertRaises(StaleVersionError):
            confirm_trip(self.conn, req(version=1))
        # 冲突不留占用
        occ = active_occupancy(self.conn, current_generation(self.conn))
        self.assertNotIn("E1", occ)

    def test_current_version_succeeds(self):
        v = current_version(self.conn)
        t = confirm_trip(self.conn, req(version=v))
        self.assertEqual(t.version_id, v)

    def test_no_version_check_when_omitted(self):
        publish_closures(self.conn, {"closures": []})
        t = confirm_trip(self.conn, req())  # 不带 data_version
        self.assertEqual(t.version_id, current_version(self.conn))


class SchemaValidationTests(unittest.TestCase):
    def test_capacity_must_be_positive_integer(self):
        for bad in (0, -1, 1.5, "3", True):
            with self.assertRaises(ValidationError):
                normalize_graph({
                    "nodes": ["A", "B"],
                    "edges": [{"id": "E1", "from": "A", "to": "B",
                               "travel_seconds": 60, "has_stairs": False,
                               "capacity": bad}],
                })

    def test_capacity_null_or_absent_means_unlimited(self):
        graph = normalize_graph({
            "nodes": ["A", "B"],
            "edges": [{"id": "E1", "from": "A", "to": "B",
                       "travel_seconds": 60, "has_stairs": False,
                       "capacity": None}],
        })
        self.assertIsNone(graph["edges"][0]["capacity"])

    def test_trip_request_party_size(self):
        nodes = {"A", "B"}
        for bad in (0, -1, 1.0, "2", True, None):
            with self.assertRaises(ValidationError):
                normalize_trip_request(
                    {"origin": "A", "destination": "B",
                     "departure_time": 1000, "party_size": bad}, nodes)

    def test_trip_request_requires_fields_and_known_nodes(self):
        with self.assertRaises(ValidationError):
            normalize_trip_request(
                {"origin": "A", "destination": "B",
                 "departure_time": 1000}, {"A", "B"})
        with self.assertRaises(ValidationError):
            normalize_trip_request(
                {"origin": "A", "destination": "X",
                 "departure_time": 1000, "party_size": 1}, {"A", "B"})

    def test_expected_version_must_be_positive_int(self):
        with self.assertRaises(ValidationError):
            normalize_trip_request(
                {"origin": "A", "destination": "B", "departure_time": 1000,
                 "party_size": 1, "data_version": 0}, {"A", "B"})
        with self.assertRaises(ValidationError):
            normalize_trip_request(
                {"origin": "A", "destination": "B", "departure_time": 1000,
                 "party_size": 1, "data_version": "2"}, {"A", "B"})


class CapacityPersistenceTests(TripTestBase):
    def test_capacity_round_trips_through_versions(self):
        snap = get_snapshot(self.conn)
        self.assertEqual(snap.edges[0].capacity, 1)
        # 恢复缺省容量的图
        publish_graph(self.conn, {
            "nodes": ["A", "B"],
            "edges": [{"id": "E1", "from": "A", "to": "B",
                       "travel_seconds": 60, "has_stairs": False}],
        })
        self.assertIsNone(get_snapshot(self.conn).edges[0].capacity)
        # 历史版本容量仍可读取
        self.assertEqual(get_snapshot(self.conn, 1).edges[0].capacity, 1)


if __name__ == "__main__":
    unittest.main()
