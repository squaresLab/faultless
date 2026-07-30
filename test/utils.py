"""Utilities for testing the codealign object model
"""

import unittest
from typing import Sequence

from faultless.ir import *
from faultless.c import UNARY_OPS, INFIX_OPS, parse_string_literal_content

# Doing it in this order is important so that unary - is overriden by infix -.
OP_MAPPING = UNARY_OPS.copy()
OP_MAPPING.update(INFIX_OPS)

def strlit(value: str) -> StringLiteral:
    return StringLiteral(value, parse_string_literal_content(value))

class TestIRGeneration(unittest.TestCase):    
    def assertContentsEqual(self, basic_block: BasicBlock, comparison: Sequence[VarInstruction]):
        """Determine if a basic block (in var form) has the same contents as the specified list.

        The == magic methods for many of the object types involved work based off those objects' 
        python IDs/memory locations, so that different instances of objects with the same contents
        are not considered equal. This is important for the correct interation of the objects with
        dictionaries in the algorithms included in this package; however, it necessitates a separate
        function (this one) for comparing based on content.
        
        This method ignores the subtleties of variable scoping.

        Consequences of ignoring variable scoping: some solutions which are actually not equivalent
        are considered equivalent by this method. If there are two different variables with the same
        name at different scopes in the original textual code but the basic_block instructions treat
        this as one variable, this method will not detect this error.
        """
        self.assertEqual(len(basic_block.instructions), len(comparison))

        # To make writing tests easier, allow for common operands to be written with strings and just substitute them in.
        for instruction in comparison:
            if isinstance(instruction.op, str) and instruction.op in OP_MAPPING:
                instruction.op = OP_MAPPING[instruction.op]

        def operands_equivalent(bb_operand, comp_operand):
            assert isinstance(bb_operand, VarOperand)
            assert isinstance(comp_operand, VarOperand)
            assert type(bb_operand) == type(comp_operand), f"{bb_operand}: {type(bb_operand)} != {comp_operand}: {type(comp_operand)}"
            
            if isinstance(bb_operand, Variable):
                self.assertEqual(bb_operand.name, comp_operand.name) # type: ignore
                self.assertEqual(bb_operand.type, comp_operand.type, bb_operand.name) # type: ignore 
            elif isinstance(bb_operand, Constant):
                assert bb_operand.value == comp_operand.value, f"{bb_operand.value} != {comp_operand.value}" # type: ignore
                if isinstance(bb_operand, IntegerConstant) or isinstance(bb_operand, FloatConstant):
                    assert bb_operand.type == comp_operand.type, f"Constant type mismatch: {bb_operand.type} vs {comp_operand.type}" # type: ignore
            else:
                assert isinstance(bb_operand, CType)
                assert bb_operand == comp_operand
        
        for bb_instruction, comp_instruction in zip(basic_block.instructions, comparison):
            # Check that types are valid
            assert isinstance(bb_instruction, VarInstruction)
            assert isinstance(comp_instruction, VarInstruction)
            # assert isinstance(bb_instruction.op, str)
            # assert isinstance(comp_instruction.op, str)
            assert bb_instruction.result is None or isinstance(bb_instruction.result, Variable) # Can be none for control-flow operations like if and return.
            assert comp_instruction.result is None or isinstance(comp_instruction.result, Variable) # Can be none for control-flow operations like if and return.
            
            # Function calls with Variable names will trivially compare false because variables
            # compare based on id (memory address). Thus we manually verify their equivalence here.
            if isinstance(bb_instruction.op, FunctionCall):
                assert isinstance(comp_instruction.op, FunctionCall)
                assert type(bb_instruction.op.fname) == type(comp_instruction.op.fname), f"Function names {bb_instruction.op.fname} and {comp_instruction.op.fname} have differing types: {type(bb_instruction.op.fname)} and {type(comp_instruction.op.fname)}"
                if isinstance(bb_instruction.op.fname, Variable) and isinstance(comp_instruction.op.fname, Variable):
                    assert bb_instruction.op.fname.name == comp_instruction.op.fname.name
                    assert bb_instruction.op.fname.type == comp_instruction.op.fname.type
                else:
                    assert bb_instruction.op == comp_instruction.op, f"Mismatched function call operations: {bb_instruction.op} and {comp_instruction.op}"
            else:
                assert bb_instruction.op == comp_instruction.op, f"Mismatched instruction operations: {bb_instruction.op} ({type(bb_instruction.op)}) and {comp_instruction.op} ({type(comp_instruction.op)})"
            
            assert type(bb_instruction.result) == type(comp_instruction.result)  
            if bb_instruction.result is not None:
                assert bb_instruction.result.name == comp_instruction.result.name, f"{bb_instruction.result.name} != {comp_instruction.result.name}" # type: ignore
                assert bb_instruction.result.type == comp_instruction.result.type, f"{bb_instruction.result.type} != {comp_instruction.result.type}" # type: ignore
            
            assert len(bb_instruction.operands) == len(comp_instruction.operands)
            for bb_operand, comp_operand in zip(bb_instruction.operands, comp_instruction.operands):
                operands_equivalent(bb_operand, comp_operand)



z3solver = z3.Solver()
class TestValues(unittest.TestCase):
    def z3eq(self, expr1, expr2) -> tuple[bool, str]:
        """Determine if two z3 expressions are equivalent by checking if they are
        """
        z3solver.push()
        z3solver.add(z3.Not(expr1 == expr2)) # check validity
        result = z3solver.check()
        assert result != z3.unknown, f"Solver limitation: unknown result (on {expr1 == expr2})"
        if result == z3.sat:
            model = z3solver.model()
            str1 = str(expr1)
            str1 = f"({str1})" if "!=" in str1 else str1
            str2 = str(expr2)
            str2 = f"({str2})" if "!=" in str2 else str2   
            counterexample = f"{str1} != {str2}\nCounterexample:\n" + "\n".join(f"{d.name()} = {model[d]}" for d in model.decls())
        else:
            counterexample = ""
        z3solver.pop()
        return result == z3.unsat, counterexample

    def assertValuesEqual(self, v1: Value, v2: Value, err_message_context: str = ""):
        """unittest.TestCase.assertEquals, but designed to handle Values. In particular:
        - If the the values are Addressable and have Variable base addresses, compare them by type and name (because Variables' __eq__ method works by ID.)
        - Compare z3 expressions by checking validity on a solver.
        """
        suffix = f" at {err_message_context}" if err_message_context else ""

        if isinstance(v1, CompoundValue) and isinstance(v2, CompoundValue):
            self.assertEqual(v1.type, v2.type, f"CompoundValue CType mismatch{suffix}: {v1.type} != {v2.type}")
            left_fields = {offset: val for offset, val in v1}
            right_fields = {offset: val for offset, val in v2}
            self.assertEqual(left_fields.keys(), right_fields.keys(), f"CompoundValue field offsets mismatch{suffix}: {set(left_fields)} != {set(right_fields)}")
            for offset in left_fields:
                field_context = f"offset {offset}" if not err_message_context else f"{err_message_context}, offset {offset}"
                self.assertValuesEqual(left_fields[offset], right_fields[offset], field_context)
            return
        
        # Place this below the check for CompopundValue to account for LazyCompoundValues.
        self.assertEqual(type(v1), type(v2), f"Value type mismatch{suffix}: {type(v1)} != {type(v2)}")

        if isinstance(v1, AddressableValue) and isinstance(v2, AddressableValue):
            if isinstance(v1.base_address, Variable):
                self.assertEqual(type(v1.base_address), type(v2.base_address))
                self.assertEqual(v1.base_address.name, v2.base_address.name) # type: ignore
            else:
                self.assertIsInstance(v2.base_address, Symbol)
                self.assertEqual(v1.base_address.type, v2.base_address.type)
                self.assertEqual(v1.base_address.name, v2.base_address.name)
                self.assertTrue(self.z3eq(v1.base_address.symvar, v2.base_address.symvar)[0], f"Base address mismatch{suffix}: {v1.base_address} != {v2.base_address}") # type: ignore -- unittest.TestCase's assertIsInstance is not recognized by the typechecker.
                self.assertEqual(v1.base_address.is_induction_var, v2.base_address.is_induction_var) # type: ignore
            self.assertEqual(v1.fields, v2.fields)
        self.assertEqual(v1.type, v2.type)
        result, counterexample = self.z3eq(v1.expr, v2.expr)
        self.assertTrue(result, f"Value mismatch{suffix}: {counterexample}")

    def assertOffsetsEqual(self, candidate: Offset, reference: Offset):
        self.assertEqual(type(candidate), type(reference))
        if isinstance(candidate, InductiveOffset) and isinstance(reference, InductiveOffset):
            self.assertTrue(*self.z3eq(candidate.induction_var, reference.induction_var))
            self.assertTrue(*self.z3eq(candidate.base_case, reference.base_case))
            self.assertTrue(*self.z3eq(candidate.update, reference.update))
        self.assertTrue(*self.z3eq(candidate.index, reference.index))
        self.assertTrue(*self.z3eq(candidate._condition, reference._condition))
        self.assertTrue(*self.z3eq(candidate.read_size, reference.read_size))

    def assertMemoryDAGsEqual(self, candidate: AddressSet | None, reference: AddressSet | None):
        match (candidate, reference):
            case Write(offset=co, value=cv, history=ch), Write(offset=ro, value=rv, history=rh):
                self.assertOffsetsEqual(co, ro)
                self.assertValuesEqual(cv, rv)
                self.assertMemoryDAGsEqual(ch, rh)
            case Join(condition=cc, true=ct, false=cf), Join(condition=rc, true=rt, false=rf):
                self.z3eq(cc, rc)
                self.assertMemoryDAGsEqual(ct, rt)
                self.assertMemoryDAGsEqual(cf, rf)
            case None, None:
                pass
            case _:
                self.assertEqual(type(candidate), type(reference))

    def assertHeapStorageEqual(self, candidate: Heap, reference: Heap):
        # Compare the keys.
        self.assertEqual(set(candidate.mapping), set(reference.mapping))
        for base_address, ref_list in reference.mapping.items():
            self.assertMemoryDAGsEqual(candidate.mapping[base_address], ref_list)
