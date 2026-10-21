"""Unit tests for inning splitting and batting lines. Run: python -m unittest scorebook.test_game"""
import unittest

from scorebook import game


def cell(result="out", numeral=None):
    return {"pa": True, "result": result, "numeral": numeral}


class SplitInnings(unittest.TestCase):
    def test_new_column_each_inning(self):
        cols = [
            (0, {"r1": cell(numeral="I"), "r2": cell(numeral="II"), "r3": cell(numeral="III")}, {}),
            (1, {"r4": cell(numeral="I"), "r5": cell(numeral="II"), "r6": cell(numeral="III")}, {}),
        ]
        pas = game.assemble(cols)
        self.assertEqual([p.inning for p in pas], [1, 1, 1, 2, 2, 2])

    def test_wrap_inside_column_is_one_inning(self):
        # inning starts at slot 8 of a 9-man order and wraps to slot 1 in the same column
        cols = [
            (0, {f"r{i}": cell() for i in range(1, 8)}, {}),
            (1, {"r8": cell(numeral="I"), "r9": cell(numeral="II"), "r1": cell(numeral="III")}, {}),
        ]
        pas = game.assemble(cols)
        self.assertEqual({p.inning for p in pas if p.column == 1}, {2})

    def test_continuation_only_when_row_is_taken(self):
        # inning 1 bats around: slots 1..9 in c0 with one out, then 1, 2 in c1
        c0 = {f"r{i}": cell("single") for i in range(1, 10)}
        c0["r9"] = cell(numeral="I")
        c1 = {"r1": cell(numeral="II"), "r2": cell(numeral="III")}
        c2 = {"r3": cell(numeral="I"), "r4": cell(numeral="II"), "r5": cell(numeral="III")}
        pas = game.assemble([(0, c0, {}), (1, c1, {}), (2, c2, {})])
        self.assertEqual({p.inning for p in pas if p.column == 1}, {1})
        self.assertEqual({p.inning for p in pas if p.column == 2}, {2})

    def test_inning_closed_without_third_out_numeral(self):
        # two outs marked, then the next column starts on a free row -> new inning
        c0 = {"r1": cell(numeral="I"), "r2": cell("walk"), "r3": cell(numeral="II")}
        c1 = {"r4": cell(numeral="I"), "r5": cell(numeral="II"), "r6": cell(numeral="III")}
        pas = game.assemble([(0, c0, {}), (1, c1, {})])
        self.assertEqual({p.inning for p in pas if p.column == 1}, {2})


class Batting(unittest.TestCase):
    def test_line(self):
        cols = [(0, {"r1": cell("double"), "r2": cell("walk"), "r3": cell(numeral="I")},
                 {"r1": {"UR": 2, "LL": 3}})]
        pas = game.assemble(cols)
        lines = {b.slot: b for b in game.batting(pas)}
        self.assertEqual((lines[1].PA, lines[1].AB, lines[1].H, lines[1].doubles, lines[1].R), (1, 1, 1, 1, 1))
        self.assertEqual((lines[2].PA, lines[2].AB, lines[2].BB), (1, 0, 1))
        self.assertEqual(game.line_score(pas), [1])


if __name__ == "__main__":
    unittest.main()
