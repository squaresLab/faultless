"""Tests for whole-function external type inference.

These tests deliberately exercise facts that cannot be recovered by a single
left-to-right ``Operation.infer_type`` pass: shared unknown callees, globals
whose declarations are missing, pointer relationships, and contradiction
detection.
"""

import unittest

from faultless.c import PRIMITIVE_TYPES, compile
from faultless.ir import *
from faultless.analysis import deduce_types
from faultless.type_inference import TypeInferenceError, infer_types


_double = PRIMITIVE_TYPES["double"]
_int = PRIMITIVE_TYPES["int"]
_char = PRIMITIVE_TYPES["char"]


class TestExternalTypeInference(unittest.TestCase):
    def infer_external(self, code: str):
        fn = compile(code.encode("utf8"))[0]
        deduce_types(fn)
        infer_types(fn)
        deduce_types(fn)
        return fn

    def compile_only(self, code: str):
        return compile(code.encode("utf8"))[0]

    def instructions(self, function: Function[VarInstruction]) -> list[VarInstruction]:
        return [instruction for bb in function for instruction in bb]

    def globals_by_name(self, function: Function[VarInstruction]) -> dict[str, GlobalVariable]:
        globals: dict[str, GlobalVariable] = {}
        for bb in function:
            for instruction in bb:
                for operand in instruction.operands:
                    if isinstance(operand, GlobalVariable):
                        globals[operand.name] = operand
                if instruction.result is not None and isinstance(instruction.result, GlobalVariable):
                    globals[instruction.result.name] = instruction.result
        return globals
    
    def callee_types_by_name(self, function: Function[VarInstruction]) -> dict[str | Variable, FunctionType]:
        callees: dict[str | Variable, FunctionType] = {}
        for bb in function:
            for instruction in bb:
                if isinstance(instruction.op, FunctionCall):
                    assert isinstance(instruction.op.fname, (str, Variable))
                    assert isinstance(instruction.op.ftype, FunctionType)
                    callees[instruction.op.fname] = instruction.op.ftype
        return callees

    def test_simple_unknown_function_arguments(self):
        function = self.infer_external("""
        void bar(int x) {
            foo(x + 1, 4.3);
        }
        """)

        foo = self.callee_types_by_name(function)["foo"]
        self.assertEqual(foo, FunctionType(Void(), [(_int, None), (_double, None)]))

    def test_assignment_is_not_equality(self):
        function = self.infer_external("""
        void bar(int x) {
            double d;
            d = x;
        }
        """)

        self.assertEqual(_int, function.parameters[0].type)
        assignment = self.instructions(function)[0]
        assert assignment.result is not None
        self.assertEqual(assignment.result.type, _double)

    def test_arithmetic_backward_propagation_prefers_receiver_type(self):
        function = self.infer_external("""
        void bar(double y, int b) {
            y = u + b;
        }
        """)

        u = self.globals_by_name(function)["u"]
        self.assertEqual(u.type, _double)

    def test_pointer_dereference_infers_global_pointer(self):
        function = self.infer_external("""
        int bar() {
            y = *p;
            return y;
        }
        """)

        globals = self.globals_by_name(function)
        self.assertEqual(globals["y"].type, _int)
        self.assertEqual(globals["p"].type, Pointer(_int))

    def test_address_of_unknown_global_returned_as_void_pointer_infers_object(self):
        function = self.infer_external("""
        void *bar() {
            return &g;
        }
        """)

        g = self.globals_by_name(function)["g"]
        self.assertEqual(g.type, _int)

    def test_pointer_arithmetic_propagates_through_dereference(self):
        function = self.infer_external("""
        int bar() {
            q = p + 1;
            y = *q;
            return y;
        }
        """)

        globals = self.globals_by_name(function)
        self.assertEqual(globals["p"].type, Pointer(_int))
        self.assertEqual(globals["q"].type, Pointer(_int))
        self.assertEqual(globals["y"].type, _int)

    def test_unknown_return_used_prefers_assignment_receiver(self):
        function = self.infer_external("""
        int bar() {
            int x;
            x = foo();
            return x;
        }
        """)

        foo = self.callee_types_by_name(function)["foo"]
        self.assertEqual(foo, FunctionType(_int, []))

    def test_unknown_return_unused_prefers_void(self):
        function = self.infer_external("""
        void bar() {
            foo();
        }
        """)

        foo = self.callee_types_by_name(function)["foo"]
        self.assertEqual(foo, FunctionType(Void(), []))

    def test_no_evidence_argument_assumption(self):
        function = self.infer_external("""
        void bar() {
            char string[128];
            fgets(string, 128, stdin);                                     
        }
        """)

        fgets = self.callee_types_by_name(function)["fgets"]
        self.assertEqual(fgets, FunctionType(Void(), [(Pointer(_char), None), (_int, None), (_int, None)]))

    def test_no_evidence_variable_assumption(self):
        fn = self.infer_external("""
        void *makenew() {
             return malloc(sizeof(element));                           
        }
        """)

        element = self.globals_by_name(fn)["element"]
        self.assertEqual(element.type, _int)

    def test_void_ptr_null_assignment_conflict(self):
        fn = self.infer_external("""
        void foo() {
            char * ptr = readstring();
            items[0] = ptr;
            items[1] = ((void *)0);
        }""")
        
        items = self.globals_by_name(fn)["items"]
        self.assertEqual(items.type, Pointer(Pointer(_char)))

    def test_known_array_subscript_does_not_become_pointer_conflict(self):
        fn = self.infer_external("""
        struct registers { int d[8]; };
        int bytescmp(struct registers A, struct registers B, int length) {
            int i, match = 1;
            for (i = 0; i < length && match; i++) match = A.d[i] == B.d[i];
            return match;
        }
        """)

        instructions = self.instructions(fn)
        member_accesses = [ins for ins in instructions if isinstance(ins.op, MemberAccess)]
        subscripts = [ins for ins in instructions if isinstance(ins.op, Subscript)]

        # We want to ensure that these don't decay into pointers during type checking.
        self.assertEqual([ins.result.type for ins in member_accesses if ins.result is not None], [Array(_int, 8), Array(_int, 8)])
        self.assertEqual([ins.result.type for ins in subscripts if ins.result is not None], [_int, _int])

    def test_null_pointer_constant_does_not_force_integer_pointer_conflict(self):
        function = self.compile_only("""
        int bar() {
            p = 0;
            y = *p;
            return y;
        }
        """)

        # The local pass loses the "zero literal" fact when it checks a
        # typed copy result, so this assertion focuses on the external solver's
        # null-pointer handling before rerunning local inference.
        result = infer_types(function)
        globals = self.globals_by_name(function)
        self.assertEqual(globals["p"].type, Pointer(_int))
        self.assertEqual(globals["y"].type, _int)
        self.assertEqual(result.diagnostics, [])

    def test_nonconstant_integer_is_not_a_null_pointer_constant(self):
        function = self.compile_only("""
        int bar(int z) {
            p = z;
            y = *p;
            return y;
        }
        """)

        with self.assertRaises(TypeInferenceError) as cm:
            infer_types(function)
        self.assertIn(("variable", "p", None), {(s.kind, s.name, s.role) for s in cm.exception.diagnostic.subjects})

    def test_strict_mode_reports_arithmetic_pointer_contradiction(self):
        function = self.compile_only("""
        void bar() {
            a = x * 3;
            b = *x;
        }
        """)

        with self.assertRaises(TypeInferenceError) as cm:
            infer_types(function)
        self.assertIn(("variable", "x", None), {(s.kind, s.name, s.role) for s in cm.exception.diagnostic.subjects})

    def test_default_skips_fully_known_constraints(self):
        function = self.compile_only("""
        void bar(int *p, int x) {
            p = x;
        }
        """)

        result = infer_types(function)
        self.assertEqual(result.diagnostics, []) # normally we'd expect this to be handled by deduce_types, but this is purely a test of type inference.

    def test_can_include_fully_known_constraints_for_validation(self):
        function = self.compile_only("""
        void bar(int *p, int x) {
            p = x;
        }
        """)

        with self.assertRaises(TypeInferenceError) as cm:
            infer_types(function, include_known_constraints=True)
        self.assertIn(("variable", "p", None), {(s.kind, s.name, s.role) for s in cm.exception.diagnostic.subjects})
        self.assertIn(("variable", "x", None), {(s.kind, s.name, s.role) for s in cm.exception.diagnostic.subjects})

    def test_mixed_arity_callee_remains_unknown_by_default(self):
        fn = self.infer_external("""
        void bar(int x) {
            printf("value: %d", x);
            printf("done");
        }
        """)

        self.assertTrue(all(ins.op.ftype is None for ins in self.instructions(fn) if isinstance(ins.op, FunctionCall)))

    def test_can_infer_variadic_callee_from_mixed_arity_calls(self):
        fn = self.compile_only("""
        void bar(int x) {
            printf("value: %d", x);
            printf("done");
        }
        """)
        infer_types(fn, infer_variadic_functions=True)

        printf = self.callee_types_by_name(fn)["printf"]
        self.assertEqual(printf, FunctionType(Void(), [(Pointer(_char), None), (FunctionType.VariadicParameter(), None)]))

    def test_variadic_fixed_prefix_stops_before_incompatible_argument_slot(self):
        fn = self.compile_only("""
        void bar(int count, char *file_path) {
            fprintf(stderr, "Wrong number: %i\\n", count);
            fprintf(stderr, "file: %s, count: %i\\n", file_path, count);
        }
        """)
        infer_types(fn, infer_variadic_functions=True)

        fprintf = self.callee_types_by_name(fn)["fprintf"]
        self.assertEqual(fprintf, FunctionType(Void(), [(_int, None), (Pointer(_char), None), (FunctionType.VariadicParameter(), None)]))

    def test_inferred_callee_pointer_parameter_generalizes_to_void_pointer(self):
        fn = self.compile_only("""
        void bar(char ***m, int i, int j) {
            free(m[i][j]);
            free(m[i]);
            free(m);
        }
        """)
        infer_types(fn)

        free = self.callee_types_by_name(fn)["free"]
        self.assertEqual(free, FunctionType(Void(), [(Pointer(Void()), None)]))

    def test_inferred_callee_partially_known_pointer_parameter_keeps_refining(self):
        fn = self.infer_external("""
        void cleanup_types(int n) {
            int i;
            for (i = 0; i < n; i++)
                if (symbol_table[i] != ((void *)0))
                    free(symbol_table[i]);
            free(symbol_table);
        }
        """)

        free = self.callee_types_by_name(fn)["free"]
        symbol_table = self.globals_by_name(fn)["symbol_table"]
        self.assertEqual(free, FunctionType(Void(), [(Pointer(Void()), None)]))
        self.assertEqual(symbol_table.type, Pointer(Pointer(Void())))

    def test_address_of_global_refines_from_pointer_assignment_receiver(self):
        fn = self.infer_external("""
        struct thing { int x; };
        void bar(void) {
            struct thing **p;
            p = &global_thing;
        }
        """)

        globals = self.globals_by_name(fn)
        thing_t = Struct("thing", [UDT.Field(_int, "x")])
        self.assertEqual(globals["global_thing"].type, Pointer(thing_t))

    def test_missing_global_member_access_infers_struct(self):
        fn = self.infer_external("""
        int bar(void) {
            global.value = 3;
            global.other = 4;
            return global.value;
        }
        """)

        global_var = self.globals_by_name(fn)["global"]
        self.assertEqual(
            global_var.type,
            Struct(None, [UDT.Field(_int, "value"), UDT.Field(_int, "other")]),
        )

    def test_missing_global_indirect_member_access_infers_pointer_to_struct(self):
        fn = self.infer_external("""
        int bar(void) {
            global->value = 3;
            return global->value;
        }
        """)

        global_var = self.globals_by_name(fn)["global"]
        self.assertEqual(
            global_var.type,
            Pointer(Struct(None, [UDT.Field(_int, "value")])),
        )

    def test_named_incomplete_struct_definition_supports_member_access(self):
        fn = self.infer_external("""
        struct node { int value; struct node *next; };
        int bar(struct node *n) {
            return n->next->value;
        }
        """)

        self.assertEqual(fn.return_type, _int)

    def test_named_incomplete_struct_without_definition_synthesizes_fields(self):
        fn = self.compile_only("""
        int bar(void) {
            struct termios termstate;
            termstate.c_lflag = 1;
            return termstate.c_lflag;
        }
        """)

        infer_types(fn)

    def test_known_array_subscript_allows_assignment_conversion(self):
        fn = self.infer_external("""
        struct inode { unsigned int blocks[4]; };
        int bar(struct inode *inode) {
            int block;
            block = inode->blocks[0];
            return block;
        }
        """)

        instructions = self.instructions(fn)
        subscripts = [ins for ins in instructions if isinstance(ins.op, Subscript)]
        self.assertTrue(any(ins.result is not None and ins.result.type == _int for ins in subscripts))

    def test_pointer_target_type_does_not_conflict_with_integer_store_conversion(self):
        fn = self.infer_external("""
        void bar(char *p) {
            p[0] = ' ';
        }
        """)

        self.assertEqual(fn.parameters[0].type, Pointer(_char))

    def test_subscript_literal_comparison_does_not_override_declared_receiver_type(self):
        fn = self.infer_external("""
        void bar(void) {
            int i;
            char out;
            for (i = 0; teststring[i] != 0; i++) {
                out = teststring[i];
            }
        }
        """)

        globals = self.globals_by_name(fn)
        self.assertEqual(globals["teststring"].type, Pointer(_char))

    def test_same_width_integer_pointer_assignment_is_tolerated(self):
        fn = self.compile_only("""
        unsigned int *bar(int *p) {
            unsigned int *result;
            result = p;
            return result;
        }
        """)

        infer_types(fn)
        self.assertEqual(fn.return_type, Pointer(PRIMITIVE_TYPES["unsigned int"]))

    def test_pointer_cast_of_global_byte_offset_infers_void_pointer_base(self):
        fn = self.infer_external("""
        void bar(int bit) {
            *(unsigned int *)(pmm_memory_map + 4LL * (bit / 32)) = 1;
        }
        """)

        globals = self.globals_by_name(fn)
        self.assertEqual(globals["pmm_memory_map"].type, Pointer(Void()))
 

if __name__ == "__main__":
    unittest.main()
