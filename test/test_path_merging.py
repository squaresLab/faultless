"""Test the path merging algorithm for combining multiple paths at join points.
"""

from .utils import TestValues

from faultless.ir import PathCondition, ComponentwisePathT, BasicBlock, NonTautologicalGroup, z3repr, INTEGER

import z3

# For convenience for writing the oracle solutions
_int = INTEGER
Merge = PathCondition.MergeTree

class TestPathMerging(TestValues): # Extend TestValues for z3eq
    def assertPathsEqual(self, candidate: PathCondition, reference: ComponentwisePathT):
        self.assertSetEqual(set(candidate.components), set(reference))
        
        for bb in reference.keys():
            cand = candidate.components[bb]
            ref = reference[bb]
            self.assertTrue(*self.z3eq(cand[0], ref[0]))
            self.assertEqual(cand[1], ref[1])

    def assertMergeTreesEqual(self, candidate: PathCondition.MergeTree | int, reference: PathCondition.MergeTree | int):
        if isinstance(candidate, PathCondition.MergeTree) and isinstance(reference, PathCondition.MergeTree):
            self.assertTrue(*self.z3eq(candidate.decision, reference.decision))
            self.assertEqual(candidate.decision_block, reference.decision_block)
            self.assertMergeTreesEqual(candidate.true, reference.true)
            self.assertMergeTreesEqual(candidate.false, reference.false)
            self.assertPathsEqual(PathCondition(candidate.true_asserts), reference.true_asserts)
            self.assertPathsEqual(PathCondition(candidate.false_asserts), reference.false_asserts)
        else:
            self.assertEqual(candidate, reference)

    def test_simple_if_else_join(self):
        """if (x > 0) { /* ... */} else { /* ... */ } <merge>
        """

        if_block = BasicBlock([], [], []) # For path-testing purposes the contents of the basic block don't matter.

        x = z3repr((_int, "x"))

        start = PathCondition()
        if_branch, else_branch = start.branch(if_block, x > 0)

        p, merge = PathCondition.merge([if_branch, else_branch])
        self.assertPathsEqual(p, {})
        self.assertMergeTreesEqual(merge, Merge(x > 0, if_block, 0, {}, 1, {}))

    def test_if_if_else_tri_join(self):
        """if (x > 0) { if (y > 0) /* ... */ else /* ... */ } <merge>"""
        
        outer_if = BasicBlock([], [], [])
        inner_if = BasicBlock([], [], [])

        x = z3repr((_int, "x"))
        y = z3repr((_int, "y"))

        start = PathCondition()
        if_branch, else_branch = start.branch(outer_if, x > 0)
        if_if_branch, if_else_branch = if_branch.branch(inner_if, y > 0)

        p, merge = PathCondition.merge([if_if_branch, if_else_branch, else_branch])
        self.assertPathsEqual(p, {})
        self.assertMergeTreesEqual(merge, Merge(x > 0, outer_if, Merge(y > 0, inner_if, 0, {}, 1, {}), {}, 2, {}))

    def test_nested_if_both_branches_one_nested_merge(self):
        """
        if (x > 0) {
            if (x < 4) { /* ... */ } else { /* ... */ }
            <merge1>
        } else {
            if (y > 0) { /* ... */ } else { /* ... */ }
        }
        <merge2>
        """

        outer_if = BasicBlock([], [], [])
        inner_if_true = BasicBlock([], [], [])
        inner_if_false = BasicBlock([], [], [])

        x = z3repr((_int, "x"))
        y = z3repr((_int, "y"))

        start = PathCondition()
        if_branch, else_branch = start.branch(outer_if, x > 0)
        if_if_branch, if_else_branch = if_branch.branch(inner_if_true, x < 4)
        else_if_branch, else_else_branch = else_branch.branch(inner_if_false, y > 0)

        p1, merge1 = PathCondition.merge([if_if_branch, if_else_branch])
        self.assertPathsEqual(p1, {outer_if: (x > 0, True)})
        self.assertMergeTreesEqual(merge1, Merge(x < 4, inner_if_true, 0, {}, 1, {}))

        p2, merge2 = PathCondition.merge([p1, else_if_branch, else_else_branch])
        self.assertPathsEqual(p2, {})
        self.assertMergeTreesEqual(merge2, Merge(x > 0, outer_if, 0, {}, Merge(y > 0, inner_if_false, 1, {}, 2, {}), {}))

    def test_post_loop_if_else_merging(self):
        """
        int i;
        for (i = 0; i < n; ++i) { /* ... */ }
        if (n > 32) { /* ... */ } else { /* ... */ }
        <merge>
        """
        loop_head = BasicBlock([], [], [])
        if_block = BasicBlock([], [], [])

        i = z3repr((_int, "$\\phi_i"))
        y = z3repr((_int, "y"))
        n = z3repr((_int, "n"))

        start = PathCondition()
        _, post_loop = start.branch(loop_head, i < n)
        if_branch, else_branch = post_loop.branch(if_block, n > 32)

        p, merge = PathCondition.merge([if_branch, else_branch])
        self.assertPathsEqual(p, {loop_head: (i >= n, False)})
        self.assertMergeTreesEqual(merge, Merge(n > 32, if_block, 0, {}, 1, {}))

    def test_while_break(self):
        """
        while (i < n) {
            if (i == 12) { break }
        }
        <merge point>
        """
        loop_head = BasicBlock([], [], [])
        if_block = BasicBlock([], [], [])

        i = z3repr((_int, "$\\phi_i"))
        n = z3repr((_int, "n"))

        start = PathCondition()
        loop_branch, post_loop = start.branch(loop_head, i < n)
        break_branch, _ = loop_branch.branch(if_block, i == 12)

        ntg_merge = Merge(i < n, loop_head, 0, {if_block: (i == 12, True)}, 1, {})

        p, merge = PathCondition.merge([break_branch, post_loop])
        disjunction: z3.BoolRef = z3.Or(i >= n, z3.And(i < n, i == 12)) # type: ignore
        self.assertPathsEqual(p, {NonTautologicalGroup(merge): (disjunction, True)})
        self.assertMergeTreesEqual(merge, ntg_merge)

    def test_early_return_unpacking(self):
        """
        if (x > 0) {
            if (y > 0) return x;
            // ...
        }
        <merge>
        return y;
        <merge return values>
        """
        outer_if = BasicBlock([], [], [])
        inner_if = BasicBlock([], [], [])

        x = z3repr((_int, "x"))
        y = z3repr((_int, "y"))

        start = PathCondition()
        outer_true, outer_false = start.branch(outer_if, x > 0)
        inner_true, inner_false = outer_true.branch(inner_if, y > 0)

        ntg_merge = Merge(x > 0, outer_if, 0, {inner_if: (y <= 0, False)}, 1, {})
        
        main_ret, merge = PathCondition.merge([inner_false, outer_false])
        disjunction: z3.BoolRef = z3.Or(z3.And(x > 0, y <= 0), x <= 0) # type: ignore
        self.assertPathsEqual(main_ret, {NonTautologicalGroup(ntg_merge): (disjunction, True)})
        self.assertMergeTreesEqual(merge, ntg_merge)

        final_merge = Merge(x > 0, outer_if, Merge(y > 0, inner_if, 0, {}, 1, {}), {}, 1, {})
        ret_residual_path, merge = PathCondition.merge([inner_true, main_ret])
        self.assertPathsEqual(ret_residual_path, {})
        self.assertMergeTreesEqual(merge, final_merge)
        
    def test_sequential_and_nested_nontautological_group_with_return(self):
        """
        if (a) {
            while (b) {
                if (c) break;
                if (d) return x;
                // ...
            }
            <merge point 1>
            while (e) {
                if (f) break;
            }
            <merge point 2>
        } else {
            // ...
        }
        <merge point 3>
        return y;
        <merge return values>
        """

        a, b, c, d, e, f = z3.Bools('a b c d e f')
        a_block, b_block, c_block, d_block, e_block, f_block = blocks = tuple(BasicBlock([], [], []) for _ in range(6))

        # For brevity, _ indicates "not"
        start = PathCondition()
        a_path, _a = start.branch(a_block, a)
        ab, a_b = a_path.branch(b_block, b)
        abc, ab_c = ab.branch(c_block, c)
        ab_cd, ab_c_d = ab_c.branch(d_block, d)
        # <merge point 1>
        m1, merge = PathCondition.merge([abc, a_b])

        merge_1 = Merge(b, b_block, 0, {c_block: (c, True)}, 1, {})
        ntg_1 = NonTautologicalGroup(merge_1)
        disjunction_1: z3.BoolRef = z3.Or(z3.And(b, c), z3.Not(b)) # type: ignore
        self.assertPathsEqual(m1, {a_block: (a, True), ntg_1: (disjunction_1, True)})
        self.assertMergeTreesEqual(merge, merge_1)


        m1e, m1_e = m1.branch(e_block, e)
        m1ef, m1e_f = m1e.branch(f_block, f)
        # <merge point 2>
        m2, merge = PathCondition.merge([m1ef, m1_e])
        
        merge_2 = Merge(e, e_block, 0, {f_block: (f, True)}, 1, {})
        ntg_2 = NonTautologicalGroup(merge_2)
        disjunction_2: z3.BoolRef = z3.Or(z3.And(e, f), z3.Not(e)) # type: ignore
        self.assertPathsEqual(m2, {a_block: (a, True), ntg_1: (disjunction_1, True), ntg_2: (disjunction_2, True)})
        self.assertMergeTreesEqual(merge, merge_2)

        # <merge point 3>
        m3, merge = PathCondition.merge([m2, _a])

        merge_3 = Merge(a, a_block, 0, {ntg_1: (disjunction_1, True), ntg_2: (disjunction_2, True)}, 1, {})
        ntg_3 = NonTautologicalGroup(merge_3)
        disjunction_3: z3.BoolRef = z3.Or(z3.And(a, disjunction_1, disjunction_2), z3.Not(a)) # type: ignore
        self.assertPathsEqual(m3, {ntg_3: (disjunction_3, True)})
        self.assertMergeTreesEqual(merge, merge_3)

        # <merge return values>
        m4, merge = PathCondition.merge([ab_cd, m3])
        merge_4 = Merge(a, a_block, 
            Merge(b, b_block, 
                Merge(c, c_block, 
                    1, {ntg_2: (disjunction_2, True)}, 
                    0, {d_block: (d, True)}), {}, 
                1, {ntg_2: (disjunction_2, True)}), {}, 
            1, {}
        )
        ntg_4 = NonTautologicalGroup(merge_4)
        disjunction_4: z3.BoolRef = z3.Or(
            z3.Not(a), 
            z3.And(a, z3.Not(b), disjunction_2), 
            z3.And(a, b, z3.Not(c), d),
            z3.And(a, b, c, disjunction_2)
        ) # type: ignore
        self.assertPathsEqual(m4, {ntg_4: (disjunction_4, True)})
        self.assertMergeTreesEqual(merge, merge_4)
        self.assertEqual(ntg_4.blocks, blocks) # the disjunction effectively tests the path conditions but not the block recovery.
        