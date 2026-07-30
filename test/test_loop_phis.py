"""Test that loop-phi instructions execution behavior is correct.
"""

import unittest

import z3

from .utils import TestValues
from faultless.ir import *
from faultless.c import compile, PRIMITIVE_TYPES
from faultless.analysis import deduce_types, convert_to_ssa, init_loop_phi_base_cases

_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]
_char = PRIMITIVE_TYPES["char"]
_uchar = PRIMITIVE_TYPES["unsigned char"]
_int = PRIMITIVE_TYPES["int"]
_long = PRIMITIVE_TYPES["long"]
_ulong = PRIMITIVE_TYPES["unsigned long"]


class TestLoopPhiBaseCaseIdentification(unittest.TestCase):
    def pipeline(self, code: str) -> Function[SSAInstruction]:
        var_irs = compile(bytes(code, "utf8"))
        assert len(var_irs) == 1
        var_ir = var_irs[0]
        deduce_types(var_ir)
        return convert_to_ssa(var_ir)

    def assertBaseCasesEqual(self, fn: Function[SSAInstruction], oracle: dict[tuple[BasicBlock, str], int]):
        init_loop_phi_base_cases(fn)
        for bb in fn:
            for instruction in bb:
                if isinstance(instruction.op, Phi):
                    self.assertEqual(
                        instruction.op.loop_base_case, 
                        oracle.get((bb, instruction.op.variable.name), None)
                    )
    
    def test_standard_for_loop(self):
        code = """
        void init_array(int *arr) {
            for (int i = 0; i < n; ++i) { 
                arr[i] = 0;
            }
        }
        """
        
        fn = self.pipeline(code)

        oracle = {(fn.basic_blocks[1], "i"): 0}

        self.assertBaseCasesEqual(fn, oracle)

    
    def test_diagonal_nested_for_loop(self):
        code = """
        void init_matrix(int **m) {
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                   m[i][j] = 0;
                }
            }
        }
        """

        fn = self.pipeline(code)
        outer_head = fn.basic_blocks[1]
        inner_head = fn.basic_blocks[4]
        
        oracle = {(outer_head, "i"): 0, (inner_head, "j"): 0}

        self.assertBaseCasesEqual(fn, oracle)

    def test_interacting_loop_indices(self):
        code = """
        int interacting_loop_indices(int n) {
            int i;
            for (i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    i += j;
                }
            }
            return i;
        }
        """

        fn = self.pipeline(code)
        outer_head = fn.basic_blocks[1]
        inner_head = fn.basic_blocks[4]

        oracle = {(outer_head, "i"): 0, (inner_head, "i"): 0, (inner_head, "j"): 0}

        self.assertBaseCasesEqual(fn, oracle)


class TestLoopPhiExecution(TestValues):
    def test_standard_for_loop(self):
        """Test the loop-phi for a standard for loop

        for (int i = 0; i < n; ++i) { /* ... */ }
        """
        phi = Phi(Variable(_int, "i"))
        phi.loop_base_case = 0
        heap = Heap()

        bv_zero = z3.BitVecVal(0, _int.size * 8)
        i_var = z3.BitVec("\\phi_i", _int.size * 8)
        postloop_var = z3.BitVec("\\$phi_i", _int.size * 8)
        base_addr = Symbol(_int, "\\phi_i", i_var, True)
        oracle_path_condition = z3.And(i_var >= 0, i_var % 1 == 0)
        path_condition = []

        pre_1 = phi.execute([Value(_int, bv_zero)], first_exec=True, block_path_condition=path_condition, heap=heap)
        assert isinstance(pre_1[0], Value), f"Pre-executing phi instruction results in object {pre_1} with unexpected return type: {type(pre_1)}"
        self.assertValuesEqual(pre_1[0], AddressableValue(_int, i_var, base_addr, ()))
        self.assertEqual(path_condition, [])
        self.assertIsNone(pre_1[1])

        pre_2 = phi.execute([AddressableValue(_int, i_var + 1, base_addr, ())], first_exec=False, block_path_condition=path_condition, heap=heap)
        self.assertValuesEqual(pre_2[0], AddressableValue(_int, postloop_var, Symbol(_int, "\\$phi_i", postloop_var, False), ()))
        # Path condition should be generated but not actually added to block_path_condition in the argument list.
        self.assertEqual(path_condition, [])
        self.assertEqual(len(phi.path_condition), 2)
        self.z3eq(z3.And(*phi.path_condition), oracle_path_condition)
        self.assertIsNone(pre_2[1])

        main_1 = phi.execute([Value(_int, bv_zero)], first_exec=True, block_path_condition=path_condition, heap=heap)
        assert isinstance(main_1[0], Value), f"Main execution of phi instruction results in object {main_1} with unexpected return type: {type(main_1)}"
        self.assertValuesEqual(main_1[0], AddressableValue(_int, i_var, base_addr, ()))
        self.assertEqual(len(phi.path_condition), 2)
        self.z3eq(z3.And(*path_condition), oracle_path_condition)
        self.assertIsNone(main_1[1])

        main_2 = phi.execute([AddressableValue(_int, i_var + 1, base_addr, ())], first_exec=False, block_path_condition=path_condition, heap=heap)
        self.assertValuesEqual(main_2[0], AddressableValue(_int, postloop_var, Symbol(_int, "\\$phi_i", postloop_var, False), ()))
        self.assertIsNone(main_2[1])