"""容量约束下的规划引擎测试。

占用按半开区间 ``[enter, leave)`` 叠加，超过路段 ``capacity`` 时允许在
节点等待；人数超过容量永远不可承载；容量与封闭、反向通行窗、避台阶、
字典序平局等既有规则组合生效。
"""
import unittest

from app.db import ClosureRow, EdgeRow, ReverseWindowRow, Snapshot
from app.routing import NoRouteError, RouteEngine


def make_engine(nodes, edges, closures=(), reverse_windows=(), occupancy=None):
    return RouteEngine(
        Snapshot(
            version_id=1,
            nodes=list(nodes),
            edges=[EdgeRow(*e) for e in edges],
            closures=[ClosureRow(*c) for c in closures],
            reverse_windows=list(reverse_windows),
        ),
        occupancy=occupancy,
    )


def E(eid, u, v, secs, stairs=False, capacity=None):
    return (eid, u, v, secs, stairs, capacity)


class CapacityBasicTests(unittest.TestCase):
    def setUp(self):
        # S -> T，60 秒，容量 1
        self.eng = lambda occ: make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=1)], occupancy=occ
        )

    def test_no_occupancy_immediate(self):
        r = self.eng({}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.arrival_time, 60)
        self.assertEqual(r.segments[0].waiting_seconds, 0)

    def test_unlimited_capacity_ignores_occupancy(self):
        eng = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60)],
            occupancy={"E1": [(0, 100000, 50)]},
        )
        r = eng.plan("S", "T", 0, party_size=100)
        self.assertEqual(r.segments[0].enter_time, 0)

    def test_party_larger_than_capacity_is_unreachable(self):
        with self.assertRaises(NoRouteError):
            self.eng({}).plan("S", "T", 0, party_size=2)

    def test_touching_half_open_intervals_do_not_conflict(self):
        # 既有 [0,60)：本段 0 进入会重叠，须等至 60
        r = self.eng({"E1": [(0, 60, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual((r.segments[0].enter_time, r.arrival_time), (60, 120))
        # 既有 [-10,0)：0 时刻进入半开不相交，立即通行
        r = self.eng({"E1": [(-10, 0, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 0)
        # 既有 [60,120)：[0,60) 不相接冲突，立即通行
        r = self.eng({"E1": [(60, 120, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 0)

    def test_overlap_forces_wait_until_free(self):
        # 既有 [30,90)：0 进入的 [0,60) 与之重叠，等到 90
        r = self.eng({"E1": [(30, 90, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 90)
        self.assertEqual(r.arrival_time, 150)
        self.assertEqual(r.segments[0].waiting_seconds, 90)

    def test_carry_in_occupancy_blocks_at_entry(self):
        # 既有占用在就绪前开始、仍未结束：进入时刻即超额
        r = self.eng({"E1": [(0, 100, 1)]}).plan("S", "T", 30, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 100)
        self.assertEqual(r.segments[0].waiting_seconds, 70)

    def test_short_overload_mid_interval(self):
        # 拥堵只持续一小段：等到拥堵结束即可
        r = self.eng({"E1": [(30, 40, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 40)

    def test_repeated_congestion_iterates_wait(self):
        # 等到 40 后，[40,100) 又与 [50,80) 冲突，再等到 80
        r = self.eng({"E1": [(30, 40, 1), (50, 80, 1)]}).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 80)

    def test_queue_of_existing_trips(self):
        # 容量 1，三批各 60 秒首尾相接：新一批排到 180
        occ = {"E1": [(0, 60, 1), (60, 120, 1), (120, 180, 1)]}
        r = self.eng(occ).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 180)

    def test_capacity_two_shares_load(self):
        # 容量 2：已有一段 1 人占用，本段 1 人可同时使用
        eng = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=2)],
            occupancy={"E1": [(0, 60, 1)]},
        )
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 0)
        # 两段各 1 人在 [30,60) 叠加为 2，新的一段须等
        occ = {"E1": [(0, 60, 1), (30, 90, 1)]}
        r = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=2)], occupancy=occ
        ).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 60)
        # carry-in 已占满 2 人
        r = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=2)],
            occupancy={"E1": [(-100, 30, 2)]},
        ).plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 30)


class CapacityDetourTests(unittest.TestCase):
    def setUp(self):
        # S 直达 T（C，容量 1）与绕行 S->X->T（不限）
        self.graph = [
            E("C", "S", "T", 10, capacity=1),
            E("D", "S", "X", 10),
            E("E", "X", "T", 10),
        ]

    def test_takes_detour_when_direct_full(self):
        eng = make_engine(["S", "X", "T"], self.graph,
                          occupancy={"C": [(0, 100, 1)]})
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual([s.edge_id for s in r.segments], ["D", "E"])
        self.assertEqual(r.arrival_time, 20)

    def test_waits_when_detour_is_slower(self):
        # 绕行 300 秒、等待直达 50 秒：最早到达选择等待
        graph = [
            E("C", "S", "T", 10, capacity=1),
            E("D", "S", "X", 150),
            E("E", "X", "T", 150),
        ]
        eng = make_engine(["S", "X", "T"], graph,
                          occupancy={"C": [(0, 50, 1)]})
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual([s.edge_id for s in r.segments], ["C"])
        self.assertEqual(r.segments[0].enter_time, 50)
        self.assertEqual(r.arrival_time, 60)

    def test_multihop_capacity_wait_cascades(self):
        graph = [E("A", "S", "M", 50, capacity=1),
                 E("B", "M", "T", 50, capacity=1)]
        eng = make_engine(["S", "M", "T"], graph,
                          occupancy={"A": [(0, 100, 1)]})
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual([(s.edge_id, s.enter_time, s.leave_time)
                          for s in r.segments],
                         [("A", 100, 150), ("B", 150, 200)])


class CapacityWithClosureTests(unittest.TestCase):
    def test_capacity_and_closure_wait_combine(self):
        # 边 60 秒；封闭 [90,120)，既有占用 [120,180)。
        # 0 就绪：[0,60) 无封闭冲突但与占用不相交 → 可立即通行，
        # 故改为占用 [0,60) + 封闭 [60,90)：先等到容量释放 60，
        # 60 进入会撞封闭 [60,90) → 90；[90,150) 与封闭 [60,90)
        # 半开不相接，且无占用 → 90 进入。
        eng = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=1)],
            closures=[("E1", 60, 90)],
            occupancy={"E1": [(0, 60, 1)]},
        )
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 90)
        self.assertEqual(r.arrival_time, 150)

    def test_closure_wait_must_recheck_capacity(self):
        # 就绪即遇封闭 [0,90)，90 进入时容量被 [90,150) 占用 → 等 150
        eng = make_engine(
            ["S", "T"], [E("E1", "S", "T", 60, capacity=1)],
            closures=[("E1", 0, 90)],
            occupancy={"E1": [(90, 150, 1)]},
        )
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 150)


class CapacityWithReverseWindowTests(unittest.TestCase):
    def _engine(self, occupancy):
        return make_engine(
            ["A", "B"], [E("E1", "A", "B", 60, capacity=1)],
            reverse_windows=[ReverseWindowRow("E1", 100, 200)],
            occupancy=occupancy,
        )

    def test_reverse_capacity_wait_stays_in_window(self):
        # B 反向，窗 [100,200)，占用 [100,160) → 160 进入，160 仍在窗内
        r = self._engine({"E1": [(100, 160, 1)]}).plan("B", "A", 0, party_size=1)
        self.assertEqual((r.segments[0].from_node, r.segments[0].to_node),
                         ("B", "A"))
        self.assertEqual((r.segments[0].enter_time, r.segments[0].leave_time),
                         (160, 220))

    def test_capacity_wait_past_window_end_unreachable(self):
        # 占用整个窗 [100,200)：等到 200 已无反向窗 → 不可达
        with self.assertRaises(NoRouteError):
            self._engine({"E1": [(100, 200, 1)]}).plan("B", "A", 0, party_size=1)

    def test_window_not_open_yet_then_capacity_queue(self):
        # 0 就绪等窗 100 开放；窗内已有 [100,160) 占用 → 160
        r = self._engine({"E1": [(100, 160, 1)]}).plan("B", "A", 0, party_size=1)
        self.assertEqual(r.segments[0].enter_time, 160)


class CapacityTieBreakTests(unittest.TestCase):
    def test_lex_sequence_kept_after_capacity_flattening(self):
        # 两条等长路径 E1,E3 与 E2,E4（均 10+10），容量把到达拉平，
        # 仍按全程编号序列字典序取 E1,E3
        graph = [
            E("E1", "S", "M", 10, capacity=1),
            E("E2", "S", "M", 10, capacity=1),
            E("E3", "M", "T", 10, capacity=1),
            E("E4", "M", "T", 10, capacity=1),
        ]
        eng = make_engine(["S", "M", "T"], graph, occupancy={})
        r = eng.plan("S", "T", 0, party_size=1)
        self.assertEqual([s.edge_id for s in r.segments], ["E1", "E3"])

    def test_avoid_stairs_still_applies(self):
        graph = [
            E("F", "S", "T", 10, stairs=True, capacity=100),
            E("P1", "S", "X", 10, stairs=False, capacity=2),
            E("P2", "X", "T", 10, stairs=False, capacity=2),
        ]
        eng = make_engine(["S", "X", "T"], graph, occupancy={})
        # 避台阶只能走平路；50 人超过平路容量 → 不可达（台阶路容量再大也不用）
        with self.assertRaises(NoRouteError):
            eng.plan("S", "T", 0, avoid_stairs=True, party_size=50)
        r = eng.plan("S", "T", 0, avoid_stairs=True, party_size=1)
        self.assertEqual([s.edge_id for s in r.segments], ["P1", "P2"])


if __name__ == "__main__":
    unittest.main()
