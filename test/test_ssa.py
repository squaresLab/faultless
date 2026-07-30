import unittest
from typing import Set

from faultless.c import compile, PRIMITIVE_TYPES
from faultless.analysis import convert_to_ssa, copy_propagation, deduce_types
from faultless.type_inference import infer_types
from faultless.ir import *

_int = PRIMITIVE_TYPES["int"]
_intp = Pointer(_int)

class TestPhiNodes(unittest.TestCase):
    def parse(self, code: str) -> Function:
        ir = compile(bytes(code, "utf8"))[0]
        deduce_types(ir)
        infer_types(ir)
        deduce_types(ir)
        return ir

    def test_loop(self):
        code = """
        int foo(int a, int b) {
            a = a + 1;
            do_call(a);
            my_thing(a, b);
            while (a < b){
                a = a + b;
            }
            a = a * 8;
            return a;  
        }
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert not isinstance(ssa.basic_blocks[0].instructions[0].op, Phi)
        assert isinstance(ssa.basic_blocks[1].instructions[0].op, Phi)
        assert ssa.basic_blocks[1].instructions[0].operands[0].op == Addition() and ssa.basic_blocks[1].instructions[0].operands[1].op == Addition()
        assert not isinstance(ssa.basic_blocks[2].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[3].instructions[0].op, Phi)
    
    def test_if_else(self):
        code = """
        int bar(int a) {
            int x;
            if (a) {
                x = fn(a);
            } else {
                x = fn(1-a);
                x = x + 1;
            }
            return x;
        }
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert not isinstance(ssa.basic_blocks[0].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[1].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[2].instructions[0].op, Phi)
        assert isinstance(ssa.basic_blocks[3].instructions[0].op, Phi)
        assert (ssa.basic_blocks[3].instructions[0].operands[0].op == Addition() and ssa.basic_blocks[3].instructions[0].operands[1].op == FunctionCall("fn")) or \
               (ssa.basic_blocks[3].instructions[0].operands[0].op == FunctionCall("fn") and ssa.basic_blocks[3].instructions[0].operands[1].op == Addition())
        
    def test_if_while_if(self):
        code = """
            int ifwhileif(int a, int b, int c) {
                int g = 0;
                if (a) {
                    printf("starting.");
                    while (a) {
                        if (b > 0) {
                        c = c + 1;
                        }
                        g += c;
                    }
                    printf("done.");
                } else {
                    printf("Can't loop.");
                }
            return c;
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 9
        assert not isinstance(ssa.basic_blocks[0].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[1].instructions[0].op, Phi)

        assert isinstance(ssa.basic_blocks[2].instructions[0].op, Phi)
        assert (ssa.basic_blocks[2].instructions[0].operands[0].op == Addition() and ssa.basic_blocks[2].instructions[0].operands[1].op == COPY_OP) or \
               (ssa.basic_blocks[2].instructions[0].operands[0].op == COPY_OP and ssa.basic_blocks[2].instructions[0].operands[1].op == Addition())
        
        assert isinstance(ssa.basic_blocks[2].instructions[1].op, Phi)
        assert (isinstance(ssa.basic_blocks[2].instructions[1].operands[0], Parameter) and isinstance(ssa.basic_blocks[2].instructions[1].operands[1], SSAInstruction)) or \
               (isinstance(ssa.basic_blocks[2].instructions[1].operands[0], SSAInstruction) and isinstance(ssa.basic_blocks[2].instructions[1].operands[1], Parameter))

        assert not isinstance(ssa.basic_blocks[3].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[4].instructions[0].op, Phi)

        assert isinstance(ssa.basic_blocks[5].instructions[0].op, Phi)
        assert (isinstance(ssa.basic_blocks[5].instructions[0].operands[0].op, Phi) and ssa.basic_blocks[5].instructions[0].operands[1].op == Addition()) or \
               (ssa.basic_blocks[5].instructions[0].operands[0].op == Addition() and isinstance(ssa.basic_blocks[5].instructions[0].operands[1].op, Phi))

        assert not isinstance(ssa.basic_blocks[6].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[7].instructions[0].op, Phi)
        assert isinstance(ssa.basic_blocks[8].instructions[0].op, Phi)
        assert (isinstance(ssa.basic_blocks[8].instructions[0].operands[0], Parameter) and isinstance(ssa.basic_blocks[8].instructions[0].operands[1], SSAInstruction)) or \
               (isinstance(ssa.basic_blocks[8].instructions[0].operands[0], SSAInstruction) and isinstance(ssa.basic_blocks[8].instructions[0].operands[1], Parameter))
    
    def test_uninitialized(self):
        code = """
        int event_wait(int a1){
            char v2;
            int v4;
            do {
                v4 = 0;
                v4 = event_translate(a1, &v2);
            }
            while (v4);
            return 1;
        }
        """

        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert len(ssa.basic_blocks[0].instructions) == 0
        # The only reference to the variable v2 in the SSA-form code should be within the Uninitialized constant itself
        assert ssa.basic_blocks[2].instructions[1].operands[0] == Uninitialized(ssa.basic_blocks[2].instructions[1].operands[0].value) # type: ignore
        assert not isinstance(ssa.basic_blocks[1].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[2].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[3].instructions[0].op, Phi)

    def test_uninitialized_path(self):
        code = """
        int foo(int x) {
            int y;
            if (x) {
                y = myfun();
            }
            return y;
        }
        """

        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 3
        assert not isinstance(ssa.basic_blocks[0].instructions[0].op, Phi)
        assert not isinstance(ssa.basic_blocks[1].instructions[0].op, Phi)
        assert isinstance(ssa.basic_blocks[2].instructions[0].op, Phi)
        assert isinstance(ssa.basic_blocks[2].instructions[0].operands[0], Uninitialized) and isinstance(ssa.basic_blocks[2].instructions[0].operands[1], SSAInstruction)
    
class TestCopyPropagation(unittest.TestCase):
    def parse(self, code: str) -> Function:
        ir = compile(bytes(code, "utf8"))[0]
        deduce_types(ir)
        infer_types(ir)
        deduce_types(ir)
        return ir

    def assertSSAEqual(self, generated: Function, reference: list[list[SSAInstruction]]):
        """Compare SSA form of instructions. Does not explicitly consider control flow.
        """
        explored: Set[SSAInstruction] = set()

        def assertOpsEqual(gen: SSAInstruction, ref: SSAInstruction):
            nonlocal explored

            if gen in explored:
                return # prevent infinate loops in cycles of phi nodes

            # This currently does not account for function calls with Variable names
            assert gen.op == ref.op, f"Differing operations: {gen.op} ({type(gen.op)}) and {ref.op} ({type(ref.op)})"
            explored.add(gen)
            
            assert len(gen.operands) == len(ref.operands)
            for goperand, roperand in zip(gen.operands, ref.operands):
                if isinstance(goperand, SSAInstruction):
                    assert isinstance(roperand, SSAInstruction)
                    assertOpsEqual(goperand, roperand) 
                elif isinstance(goperand, Parameter):
                    assert isinstance(roperand, Parameter)
                    assert goperand.name == roperand.name
                elif isinstance(goperand, GlobalVariable):
                    assert isinstance(roperand, GlobalVariable)
                    assert goperand.name == roperand.name
                elif isinstance(goperand, Constant):
                    assert isinstance(roperand, Constant)
                    assert goperand.value == roperand.value
                else:
                    assert isinstance(roperand, Uninitialized)
                    assert isinstance(goperand, Uninitialized)

        for gen_block, ref_instructions in zip(generated.basic_blocks, reference):
            assert len(gen_block.instructions) == len(ref_instructions)
            for gen_op, ref_op in zip(gen_block.instructions, ref_instructions):
                assertOpsEqual(gen_op, ref_op)
    
    def test_increment(self):
        code = """
        void foo() {
            int i = 0;
            i++;
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        reference = [
            SSAInstruction(Addition(), [IntegerConstant(0, _int), IntegerConstant(1, _int)])
        ]

        self.assertSSAEqual(ssa, [reference])

    def test_for_loop(self):
        code = """
        void foo(int * arr, int len) {
            for (int i = 0; i < len; i++) {
                arr[i] = -1;
            }
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        iphi = SSAInstruction(Phi(Variable(_int, "i")), [IntegerConstant(0, _int)])
        comparison = SSAInstruction(LessThan(), [iphi, Parameter(_int, "len")])
        loop = SSAInstruction(LOOP_OP, [comparison])
        condition_block = [iphi, comparison, loop]

        increment = SSAInstruction(Addition(), [iphi, IntegerConstant(1, _int)])
        iphi.operands.append(increment)
        increment_block = [increment]

        array_access = SSAInstruction(SUBSCRIPT_OP, [Parameter(_intp, "arr"), iphi])
        array_store = SSAInstruction(STORE_OP, [array_access, IntegerConstant(-1, _int)])
        body_block = [array_access, array_store]

        self.assertSSAEqual(ssa, [[], condition_block, increment_block, body_block])
    
    def test_copy_chain(self):
        code = """
        int foo(int x) {
            int y = x;
            int z = y;
            int w = z;
            if (x) {
                w = w + 5;
            } else {
                w = -1;
            }
            return w;
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        if_op = SSAInstruction(IF_OP, [Parameter(_int, "x")])
        if_condition = [if_op]

        update = SSAInstruction(Addition(), [Parameter(_int, "x"), IntegerConstant(5, _int)])
        if_body = [update]

        phi = SSAInstruction(Phi(Variable(_int, "w")), [update, IntegerConstant(-1, _int)])
        return_op = SSAInstruction(RETURN_OP, [phi])
        post_if = [phi, return_op]

        self.assertSSAEqual(ssa, [if_condition, if_body, [], post_if])
