import unittest

from faultless.c import compile, integer_literal_components, parse_int, parse_integer_literal, PRIMITIVE_TYPES
from faultless.ir import *
from .utils import TestIRGeneration, strlit

# For convenience because we use these types a lot.
_char = PRIMITIVE_TYPES["char"]
_int = PRIMITIVE_TYPES["int"]
_uint = PRIMITIVE_TYPES["unsigned int"]
_long = PRIMITIVE_TYPES["long"]
_ulong = PRIMITIVE_TYPES["unsigned long"]
_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]

_charp = Pointer(_char)
_intp = Pointer(_int)
_voidp = Pointer(PRIMITIVE_TYPES["void"])
_unk = UnknownType()


# Test that IR is formed correctly.
class TestCIRGeneration(TestIRGeneration):
    def parse(self, code: str, short_circuit: bool = True) -> Function:
        return compile(bytes(code, "utf8"), short_circuit_logical_ops=short_circuit)[0]
    
    def test_binary_op(self):
        code = """
        int foo(int x, int y) {
            int z = x + y;
        }
        """
        
        fn = self.parse(code)
        
        correct = [
            VarInstruction(Addition(), Variable(_int, "z"), [Parameter(_int, "x"), Parameter(_int, "y")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_unary_op(self):
        code = """
        int foo(int x) {
            int z = !x;
        }
        """

        fn = self.parse(code)

        correct = [
           VarInstruction(LogicalNot(), Variable(_int, "z"), [Parameter(_int, "x")]) 
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_function_call(self):
        code = """
        int foo(int x, int y) {
            int z = bar(x, y, gbl, 3);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall("bar"), Variable(_int, "z"), [Parameter(_int, "x"), Parameter(_int, "y"), GlobalVariable(_unk, "gbl"), IntegerConstant(3, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_function_pointer_call(self):
        code = """
        int foo(int x, void * fnptr) {
            fnptr(1);
            (*fnptr)(2);
            get_ptr(x)(3);
            (*get_ptr(x))(4);
            (*ident)(5, 6, 7);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall(Parameter(_voidp, "fnptr")), Variable(_unk, "t0"), [IntegerConstant(1, _int)]),
            VarInstruction(FunctionCall(Parameter(_voidp, "fnptr")), Variable(_unk, "t1"), [IntegerConstant(2, _int)]),
            VarInstruction(FunctionCall("get_ptr"), Variable(_unk, "t2"), [Parameter(_int, "x")]),
            VarInstruction(FunctionCall(Variable(_unk, "t2")), Variable(_unk, "t3"), [IntegerConstant(3, _int)]),
            VarInstruction(FunctionCall("get_ptr"), Variable(_unk, "t4"), [Parameter(_int, "x")]),
            VarInstruction(FunctionCall(Variable(_unk, "t4")), Variable(_unk, "t5"), [IntegerConstant(4, _int)]),
            VarInstruction(FunctionCall(GlobalVariable(_unk, "ident")), Variable(_unk, "t6"), [IntegerConstant(5, _int), IntegerConstant(6, _int), IntegerConstant(7, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_pointer_dereference(self):
        code = """
        int foo(int * x) {
            int z = *x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Dereference(), Variable(_int, "z"), [Parameter(_intp, "x")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_struct_field_access(self):
        code = """
        struct node { int value; struct node * left; struct node * right; };
        int foo(struct node * tree, struct node current) {
            struct node * r = tree->right;
            struct node * l = current.left;
        }
        """

        fn = self.parse(code)

        inode = IncompleteStruct("node")
        node = Struct("node", [UDT.Field(_int, "value"), UDT.Field(Pointer(inode), "left"), UDT.Field(Pointer(inode), "right")])
        inode.full_definition = node
        nodep = Pointer(node)
        correct = [
            VarInstruction(MemberAccess(True), Variable(nodep, "r"), [Parameter(nodep, "tree"), Field("right")]),
            VarInstruction(MemberAccess(False), Variable(nodep, "l"), [Parameter(node, "current"), Field("left")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_cast_expression(self):
        code = """
        int foo(long x) {
            (int)x;
            (long *)x;
            (struct { int a; int b; })x;
            (struct point[1])x;
        }
        """

        fn = self.parse(code)

        anon_struct_t = Struct(None, [UDT.Field(_int, "a"), UDT.Field(_int, "b")])
        point_array_t = Array(IncompleteStruct("point"), 1)

        correct = [
            VarInstruction(CAST_OP, Variable(_unk, "t0"), [_int, Parameter(_long, "x")]),
            VarInstruction(CAST_OP, Variable(_unk, "t1"), [Pointer(_long), Parameter(_long, "x")]),
            VarInstruction(CAST_OP, Variable(_unk, "t2"), [anon_struct_t, Parameter(_long, "x")]),
            VarInstruction(CAST_OP, Variable(_unk, "t3"), [point_array_t, Parameter(_long, "x")]),
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_fastcall_function_pointer_cast_expression(self):
        code = """
        typedef long __int64;
        typedef int _DWORD;
        typedef unsigned long _QWORD;

        long foo(void *callback) {
            return func(
                (__int64 (__fastcall *)(_DWORD *, _QWORD))callback
            );
        }
        """

        fn = self.parse(code)
        callback_t = Pointer(FunctionType(_long, [(Pointer(_int), None), (_ulong, None),],))

        correct = [
            VarInstruction(CAST_OP, Variable(_unk, "t0"), [callback_t, Parameter(_voidp, "callback")]),
            VarInstruction(FunctionCall("func"), Variable(_unk, "t1"), [Variable(_unk, "t0")]),
            VarInstruction(RETURN_OP, None, [Variable(_unk, "t1")]),
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_subscript_expression(self):
        code = """
        void foo(int * arr) {
            int x = arr[2];
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(SUBSCRIPT_OP, Variable(_int, "x"), [Parameter(_intp, "arr"), IntegerConstant(2, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_sizeof(self):
        code = """
        void foo(int x) {
            x = sizeof(x);
            x = sizeof(struct node);
            x = sizeof(struct node[12])
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(SIZEOF_OP, Parameter(_int, "x"), [Parameter(_int, "x")]),
            VarInstruction(SIZEOF_OP, Parameter(_int, "x"), [IncompleteStruct("node")]),
            VarInstruction(SIZEOF_OP, Parameter(_int, "x"), [Array(IncompleteStruct("node"), 12)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_comma_operator(self):
        code = """
        void foo(int x, void * fnptr) {
            while(next(), read()) {};
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall("next"), Variable(_unk, "t0"), []),
            VarInstruction(FunctionCall("read"), Variable(_unk, "t1"), []),
            VarInstruction(LOOP_OP, None, [Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], correct)
    
    def test_initializer_list(self):
        code = """
        void foo(int x) {
            int arr[4] = {0, x + 2, 3, fn(1)};
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Addition(), Variable(_unk, "t0"), [Parameter(_int, "x"), IntegerConstant(2, _int)]),
            VarInstruction(FunctionCall("fn"), Variable(_unk, "t1"), [IntegerConstant(1, _int)]),
            VarInstruction(Initializer(Array(_int, 4)), Variable(Array(_int, 4), "arr"), [IntegerConstant(0, _int), Variable(_unk, "t0"), IntegerConstant(3, _int), Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_implicit_size_array_initializer_list(self):
        code = """
        void foo(int x) {
            int arr[] = {0, x + 2, 3, fn(1)};
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Addition(), Variable(_unk, "t0"), [Parameter(_int, "x"), IntegerConstant(2, _int)]),
            VarInstruction(FunctionCall("fn"), Variable(_unk, "t1"), [IntegerConstant(1, _int)]),
            VarInstruction(Initializer(Array(_int, 4)), Variable(Array(_int, 4), "arr"), [IntegerConstant(0, _int), Variable(_unk, "t0"), IntegerConstant(3, _int), Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_implicit_size_array_string_initializer(self):
        code = """
        void foo() {
            char str[] = "abc";
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(Array(_char, 4), "str"), [strlit("abc")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_nested_initializer_list(self):
        code = """
        union ctoi { int val; char cs[4]; };
        union ctoi foo() {
            union ctoi myunion = { .cs={'a', 'b', 'c', '\\0'} };
            int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
        }
        """

        fn = self.parse(code)

        ctoi = Union("ctoi", [UDT.Field(_int, "val"), UDT.Field(Array(_char, 4), "cs")])

        correct = [
            VarInstruction(Initializer(Array(_char, 4)), Variable(Array(_char, 4), "t0"), [CharLiteral(ord('a')), CharLiteral(ord('b')), CharLiteral(ord('c')), CharLiteral(0)]),
            VarInstruction(Initializer(ctoi, ["cs"]), Variable(ctoi, "myunion"), [Variable(Array(_char, 4), "t0")]),
            VarInstruction(Initializer(Array(_int, 3)), Variable(Array(_int, 3), "t1"), [IntegerConstant(1, _int), IntegerConstant(2, _int), IntegerConstant(3, _int)]),
            VarInstruction(Initializer(Array(_int, 3)), Variable(Array(_int, 3), "t2"), [IntegerConstant(4, _int), IntegerConstant(5, _int), IntegerConstant(6, _int)]),
            VarInstruction(Initializer(Array(Array(_int, 3), 2)), Variable(Array(Array(_int, 3), 2), "matrix"), [Variable(Array(_int, 3), "t1"), Variable(Array(_int, 3), "t2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_plus_equals(self):
        code = """
        void foo(int a) {
            a += 1;
            a -= func();
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Addition(), Parameter(_int, "a"), [Parameter(_int, "a"), IntegerConstant(1, _int)]),
            VarInstruction(FunctionCall("func"), Variable(_unk, "t0"), []),
            VarInstruction(Subtraction(), Parameter(_int, "a"), [Parameter(_int, "a"), Variable(_unk, "t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_true_false_null(self):
        code = """
        int fn() {
            bar(NULL, true, TRUE, false, FALSE);
        }
        """
        
        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall("bar"), Variable(_unk, "t0"), [IntegerConstant(0, SIZE_T), IntegerConstant(1, _int), IntegerConstant(1, _int), IntegerConstant(0, _int), IntegerConstant(0, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_void_parameter(self):
        code = """
        int foo(void) {
            return 1;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(RETURN_OP, None, [IntegerConstant(1, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_multiple_assignment(self):
        code = """
        void foo(int x) {
           int a;
           int b;
           a = b = x;
           a = b = x % 7;
           a *= b += x;
           a >>= b -= x / 2;
           int c;
           a = b = c = x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_int, "b"), [Parameter(_int, "x")]),
            VarInstruction(COPY_OP, Variable(_int, "a"), [Variable(_int, "b")]),
            VarInstruction(ModulusDivision(), Variable(_int, "b"), [Parameter(_int, "x"), IntegerConstant(7, _int)]),
            VarInstruction(COPY_OP, Variable(_int, "a"), [Variable(_int, "b")]),
            VarInstruction(Addition(), Variable(_int, "b"), [Variable(_int, "b"), Parameter(_int, "x")]),
            VarInstruction(Multiplication(), Variable(_int, "a"), [Variable(_int, "a"), Variable(_int, "b")]),
            VarInstruction(Division(), Variable(_unk, "t0"), [Parameter(_int, "x"), IntegerConstant(2, _int)]),
            VarInstruction(Subtraction(), Variable(_int, "b"), [Variable(_int, "b"), Variable(_unk, "t0")]),
            VarInstruction(RightShift(), Variable(_int, "a"), [Variable(_int, "a"), Variable(_int, "b")]),
            VarInstruction(COPY_OP, Variable(_int, "c"), [Parameter(_int,"x")]),
            VarInstruction(COPY_OP, Variable(_int, "b"), [Variable(_int,"c")]),
            VarInstruction(COPY_OP, Variable(_int, "a"), [Variable(_int, "b")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_parenthesized_assignment_lhs_is_plain_assignment(self):
        code = """
        void foo(int x) {
            int a;
            (a) = x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_int, "a"), [Parameter(_int, "x")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_multiple_declarations_single_line(self):
        code = """
        void foo(int x) {
            int * a = 0, * b, c = x + 2, d = 4 - 3 * x;
            func(a, b, c, d);
        }
        """

        fn = self.parse(code)

        correct  = [
            VarInstruction(COPY_OP, Variable(_intp, "a"), [IntegerConstant(0, _int)]),
            VarInstruction(Addition(), Variable(_int, "c"), [Parameter(_int, "x"), IntegerConstant(2, _int)]),
            VarInstruction(Multiplication(), Variable(_unk, "t0"), [IntegerConstant(3, _int), Parameter(_int, "x")]),
            VarInstruction(Subtraction(), Variable(_int, "d"), [IntegerConstant(4, _int), Variable(_unk, "t0")]),
            # The use of Variable instead of GlobalVariable indicates that these variables were successfully declared inside the function.
            VarInstruction(FunctionCall("func"), Variable(_unk, "t1"), [Variable(_intp, "a"), Variable(_intp,"b"), Variable(_int, "c"), Variable(_int, "d")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_prefix_update(self):
        code = """
        void foo(int x, int * src, int * dst) {
            int y = ++x - 3;
            *(--dst) = *(++src)
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Addition(), Parameter(_int, "x"), [Parameter(_int, "x"), IntegerConstant(1, _int)]),
            VarInstruction(Subtraction(), Variable(_int, "y"), [Parameter(_int, "x"), IntegerConstant(3, _int)]),
            VarInstruction(Subtraction(), Parameter(_intp, "dst"), [Parameter(_intp, "dst"), IntegerConstant(1, _int)]),
            VarInstruction(Dereference(), Variable(_unk, "t0"), [Parameter(_intp, "dst")]),
            VarInstruction(Addition(), Parameter(_intp, "src"), [Parameter(_intp, "src"), IntegerConstant(1, _int)]),
            VarInstruction(Dereference(), Variable(_unk, "t1"), [Parameter(_intp, "src")]),
            VarInstruction(STORE_OP, Variable(_unk, "t0"), [Variable(_unk, "t0"), Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_logical_and_no_short_circuiting(self):
        code = """
        int foo(int a, int b) {
            return a && b;
        }
        """

        fn = self.parse(code, short_circuit=False)
        correct = [
            VarInstruction(LogicalAnd(), Variable(_unk, "t0"), [Parameter(_int, "a"), Parameter(_int, "b")]),
            VarInstruction(RETURN_OP, None, [Variable(_unk, "t0")])
        ]
        self.assertContentsEqual(fn.entry_block, correct)

    def test_logical_and_short_circuits(self):
        code = """
        int foo(int a, int b) {
            int x = a && b;
            return x;
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 4)

        entry, zero_block, rhs_block, merge_block = fn.basic_blocks

        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(entry.successors, [rhs_block, zero_block])
        self.assertEqual(entry.predecessors, [])

        self.assertContentsEqual(rhs_block, [
            VarInstruction(NotEqualTo(), Variable(_int, "t0"), [Parameter(_int, "b"), IntegerConstant(0, _int)]),
        ])
        self.assertEqual(rhs_block.predecessors, [entry])
        self.assertEqual(rhs_block.successors, [merge_block])

        self.assertContentsEqual(zero_block, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(zero_block.predecessors, [entry])
        self.assertEqual(zero_block.successors, [merge_block])

        self.assertContentsEqual(merge_block, [
            VarInstruction(COPY_OP, Variable(_int, "x"), [Variable(_int, "t0")]),
            VarInstruction(RETURN_OP, None, [Variable(_int, "x")])
        ])
        self.assertCountEqual(merge_block.predecessors, [rhs_block, zero_block])
        self.assertEqual(merge_block.successors, [])

    def test_logical_or_short_circuits(self):
        code = """
        int foo(int a, int b) {
            int x = a || b;
            return x;
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 4)

        entry = fn.entry_block
        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])

        true_block, rhs_block = entry.successors
        merge_block = true_block.successors[0]

        self.assertContentsEqual(true_block, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(1, _int)])
        ])
        self.assertEqual(true_block.predecessors, [entry])
        self.assertEqual(true_block.successors, [merge_block])

        self.assertContentsEqual(rhs_block, [
            VarInstruction(NotEqualTo(), Variable(_int, "t0"), [Parameter(_int, "b"),  IntegerConstant(0, _int)]),
        ])
        self.assertEqual(rhs_block.predecessors, [entry])
        self.assertEqual(rhs_block.successors, [merge_block])

        self.assertContentsEqual(merge_block, [
            VarInstruction(COPY_OP, Variable(_int, "x"), [Variable(_int, "t0")]),
            VarInstruction(RETURN_OP, None, [Variable(_int, "x")])
        ])
        self.assertCountEqual(merge_block.predecessors, [true_block, rhs_block])
        self.assertEqual(merge_block.successors, [])

    def test_nested_short_circuit_condition(self):
        code = """
        int f(unsigned int c) {
            if (c == '-' || (c >= '0' && c <= '9'))
                return 1;
            return 0;
        }
        """

        fn = self.parse(code, short_circuit=True)
        self.assertEqual(len(fn.basic_blocks), 9)

        entry, or_true_block, rhs_entry, rhs_false, rhs_true, rhs_merge, cond_merge, then_block, else_block = fn.basic_blocks

        self.assertContentsEqual(entry, [
            VarInstruction(EqualTo(), Variable(_unk, "t0"), [Parameter(_uint, "c"), CharLiteral(ord("-"))]),
            VarInstruction(IF_OP, None, [Variable(_unk, "t0")])
        ])
        self.assertEqual(entry.predecessors, [])
        self.assertEqual(entry.successors, [or_true_block, rhs_entry])

        self.assertContentsEqual(or_true_block, [
            VarInstruction(COPY_OP, Variable(_int, "t1"), [IntegerConstant(1, _int)])
        ])
        self.assertEqual(or_true_block.predecessors, [entry])
        self.assertEqual(or_true_block.successors, [cond_merge])

        self.assertContentsEqual(rhs_entry, [
            VarInstruction(GreaterThanOrEqualTo(), Variable(_unk, "t2"), [Parameter(_uint, "c"), CharLiteral(ord("0"))]),
            VarInstruction(IF_OP, None, [Variable(_unk, "t2")])
        ])
        self.assertEqual(rhs_entry.predecessors, [entry])
        self.assertEqual(rhs_entry.successors, [rhs_true, rhs_false])

        self.assertContentsEqual(rhs_false, [
            VarInstruction(COPY_OP, Variable(_int, "t3"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(rhs_false.predecessors, [rhs_entry])
        self.assertEqual(rhs_false.successors, [rhs_merge])

        self.assertContentsEqual(rhs_true, [
            VarInstruction(LessThanOrEqualTo(), Variable(_int, "t3"), [Parameter(_uint, "c"), CharLiteral(ord("9"))])
        ])
        self.assertEqual(rhs_true.predecessors, [rhs_entry])
        self.assertEqual(rhs_true.successors, [rhs_merge])

        self.assertContentsEqual(rhs_merge, [
            VarInstruction(NotEqualTo(), Variable(_int, "t1"), [Variable(_int, "t3"), IntegerConstant(0, _int)])
        ])
        self.assertCountEqual(rhs_merge.predecessors, [rhs_false, rhs_true])
        self.assertEqual(rhs_merge.successors, [cond_merge])

        self.assertContentsEqual(cond_merge, [
            VarInstruction(IF_OP, None, [Variable(_int, "t1")])
        ])
        self.assertCountEqual(cond_merge.predecessors, [or_true_block, rhs_merge])
        self.assertEqual(cond_merge.successors, [then_block, else_block])

        self.assertContentsEqual(then_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(1, _int)])
        ])
        self.assertEqual(then_block.predecessors, [cond_merge])
        self.assertEqual(then_block.successors, [])

        self.assertContentsEqual(else_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])
        ])
        self.assertEqual(else_block.predecessors, [cond_merge])
        self.assertEqual(else_block.successors, [])

    def test_logical_and_in_if_condition(self):
        code = """
        int foo(int a, int b) {
            if (a && b) {
                return 1;
            }
            return 0;
        }
        """

        fn = self.parse(code, short_circuit=True)
        self.assertEqual(len(fn.basic_blocks), 6)

        entry, false_block, rhs_block, cond_merge, then_block, else_block = fn.basic_blocks

        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(entry.predecessors, [])
        self.assertEqual(entry.successors, [rhs_block, false_block])

        self.assertContentsEqual(rhs_block, [
            VarInstruction(NotEqualTo(), Variable(_int, "t0"), [Parameter(_int, "b"), IntegerConstant(0, _int)])
        ])
        self.assertEqual(rhs_block.predecessors, [entry])
        self.assertEqual(rhs_block.successors, [cond_merge])

        self.assertContentsEqual(false_block, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(false_block.predecessors, [entry])
        self.assertEqual(false_block.successors, [cond_merge])

        self.assertContentsEqual(cond_merge, [
            VarInstruction(IF_OP, None, [Variable(_int, "t0")])
        ])
        self.assertSetEqual(set(cond_merge.predecessors), {rhs_block, false_block})
        self.assertEqual(cond_merge.successors, [then_block, else_block])

        self.assertContentsEqual(then_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(1, _int)])
        ])
        self.assertEqual(then_block.predecessors, [cond_merge])
        self.assertEqual(then_block.successors, []) # because of the return

        self.assertContentsEqual(else_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])
        ])
        self.assertEqual(else_block.predecessors, [cond_merge])
        self.assertEqual(else_block.successors, [])

    def test_logical_or_in_while_condition(self):
        code = """
        int foo(int a, int b) {
            while (a || b) {
                a = 0;
            }
            return a;
        }
        """

        fn = self.parse(code, short_circuit=True)
        self.assertEqual(len(fn.basic_blocks), 7)

        entry, cond_entry, short_block, rhs_block, cond_merge, body_block, exit_block = fn.basic_blocks

        self.assertContentsEqual(entry, [])
        self.assertEqual(entry.predecessors, [])
        self.assertEqual(entry.successors, [cond_entry])

        self.assertContentsEqual(cond_entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertSetEqual(set(cond_entry.predecessors), {entry, body_block})
        self.assertEqual(cond_entry.successors, [short_block, rhs_block])

        self.assertContentsEqual(short_block, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(1, _int)])
        ])
        self.assertEqual(short_block.predecessors, [cond_entry])
        self.assertEqual(short_block.successors, [cond_merge])

        self.assertContentsEqual(rhs_block, [
            VarInstruction(NotEqualTo(), Variable(_int, "t0"), [Parameter(_int, "b"), IntegerConstant(0, _int)])
        ])
        self.assertEqual(rhs_block.predecessors, [cond_entry])
        self.assertEqual(rhs_block.successors, [cond_merge])

        self.assertContentsEqual(cond_merge, [
            VarInstruction(LOOP_OP, None, [Variable(_int, "t0")])
        ])
        self.assertSetEqual(set(cond_merge.predecessors), {short_block, rhs_block})
        self.assertEqual(cond_merge.successors, [body_block, exit_block])

        self.assertContentsEqual(body_block, [
            VarInstruction(COPY_OP, Parameter(_int, "a"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(body_block.predecessors, [cond_merge])
        self.assertEqual(body_block.successors, [cond_entry])

        self.assertContentsEqual(exit_block, [
            VarInstruction(RETURN_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(exit_block.predecessors, [cond_merge])
        self.assertEqual(exit_block.successors, [])

    def test_short_circuit_for_loop_with_conditional_increment(self):
        code = """
        int foo(int *a, int n) {
            for (int i = a || b; a && i < n; i += a ? 2 : 1) {
                doit(i, &a);
            }
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 14)

        entry, init_short, init_rhs, init_merge, \
            cond_entry, cond_short, cond_rhs, cond_merge, \
            update_cond, update_cons, update_alt, update_merge, \
            body_block, exit_block = fn.basic_blocks

        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_intp, "a")])
        ])
        self.assertEqual(entry.successors, [init_short, init_rhs])

        self.assertContentsEqual(init_short, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(1, _int)])
        ])
        self.assertEqual(init_short.predecessors, [entry])
        self.assertEqual(init_short.successors, [init_merge])

        self.assertContentsEqual(init_rhs, [
            VarInstruction(NotEqualTo(), Variable(_int, "t0"), [GlobalVariable(_unk, "b"), IntegerConstant(0, _int)])
        ])
        self.assertEqual(init_rhs.predecessors, [entry])
        self.assertEqual(init_rhs.successors, [init_merge])

        self.assertContentsEqual(init_merge, [
            VarInstruction(COPY_OP, Variable(_int, "i"), [Variable(_int, "t0")])
        ])
        self.assertSetEqual(set(init_merge.predecessors), {init_rhs, init_short})
        self.assertEqual(init_merge.successors, [cond_entry])

        self.assertContentsEqual(cond_entry, [
            VarInstruction(IF_OP, None, [Parameter(_intp, "a")])
        ])
        self.assertCountEqual(cond_entry.predecessors, [init_merge, update_merge])
        self.assertEqual(cond_entry.successors, [cond_rhs, cond_short])

        self.assertContentsEqual(cond_rhs, [
            VarInstruction(LessThan(), Variable(_int, "t1"), [Variable(_int, "i"), Parameter(_int, "n")]),
        ])
        self.assertEqual(cond_rhs.predecessors, [cond_entry])
        self.assertEqual(cond_rhs.successors, [cond_merge])

        self.assertContentsEqual(cond_short, [
            VarInstruction(COPY_OP, Variable(_int, "t1"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(cond_short.predecessors, [cond_entry])
        self.assertEqual(cond_short.successors, [cond_merge])

        self.assertContentsEqual(cond_merge, [
            VarInstruction(LOOP_OP, None, [Variable(_int, "t1")])
        ])
        self.assertSetEqual(set(cond_merge.predecessors), {cond_rhs, cond_short})
        self.assertEqual(cond_merge.successors, [body_block, exit_block])

        self.assertContentsEqual(update_cond, [
            VarInstruction(IF_OP, None, [Parameter(_intp, "a")])
        ])
        self.assertEqual(update_cond.predecessors, [body_block])
        self.assertEqual(update_cond.successors, [update_cons, update_alt])

        self.assertContentsEqual(update_cons, [
            VarInstruction(COPY_OP, Variable(_unk, "t2"), [IntegerConstant(2, _int)])
        ])
        self.assertEqual(update_cons.predecessors, [update_cond])
        self.assertEqual(update_cons.successors, [update_merge])

        self.assertContentsEqual(update_alt, [
            VarInstruction(COPY_OP, Variable(_unk, "t2"), [IntegerConstant(1, _int)])
        ])
        self.assertEqual(update_alt.predecessors, [update_cond])
        self.assertEqual(update_alt.successors, [update_merge])

        self.assertContentsEqual(update_merge, [
            VarInstruction(Addition(), Variable(_int, "i"), [Variable(_int, "i"), Variable(_unk, "t2")])
        ])
        self.assertSetEqual(set(update_merge.predecessors), {update_cons, update_alt})
        self.assertEqual(update_merge.successors, [cond_entry])

        self.assertContentsEqual(body_block, [
            VarInstruction(AddressOf(), Variable(_unk, "t3"), [Parameter(_intp, "a")]),
            VarInstruction(FunctionCall("doit"), Variable(_unk, "t4"), [Variable(_int, "i"), Variable(_unk, "t3")])
        ])
        self.assertEqual(body_block.predecessors, [cond_merge])
        self.assertEqual(body_block.successors, [update_cond])

        self.assertContentsEqual(exit_block, [])
        self.assertEqual(exit_block.predecessors, [cond_merge])
        self.assertEqual(exit_block.successors, [])

    def test_do_while_with_short_circuit_condition(self):
        code = """
        int foo(int x) {
            do {
                print(x++);
            } while (x && x < 100);
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 7)

        entry_block, cond_entry, cond_short, cond_rhs, cond_merge, body_block, exit_block = fn.basic_blocks

        self.assertContentsEqual(entry_block, [])
        self.assertEqual(len(entry_block.predecessors), 0)
        self.assertEqual(entry_block.successors, [body_block])

        self.assertContentsEqual(body_block, [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [Parameter(_int, "x")]),
            VarInstruction(Addition(), Parameter(_int, "x"), [Parameter(_int, "x"), IntegerConstant(1, _int)]),
            VarInstruction(FunctionCall("print"), Variable(_unk, "t1"), [Variable(_unk, "t0")])
        ])
        self.assertSetEqual(set(body_block.predecessors), {entry_block, cond_merge})
        self.assertEqual(body_block.successors, [cond_entry])

        self.assertContentsEqual(cond_entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "x")])
        ])
        self.assertEqual(cond_entry.predecessors, [body_block])
        self.assertEqual(cond_entry.successors, [cond_rhs, cond_short])

        self.assertContentsEqual(cond_rhs, [
            VarInstruction(LessThan(), Variable(_int, "t0"), [Parameter(_int, "x"), IntegerConstant(100, _int)]),
        ])
        self.assertEqual(cond_rhs.predecessors, [cond_entry])
        self.assertEqual(cond_rhs.successors, [cond_merge])

        self.assertContentsEqual(cond_short, [
            VarInstruction(COPY_OP, Variable(_int, "t0"), [IntegerConstant(0, _int)])
        ])
        self.assertEqual(cond_short.predecessors, [cond_entry])
        self.assertEqual(cond_short.successors, [cond_merge])

        self.assertContentsEqual(cond_merge, [
            VarInstruction(LOOP_OP, None, [Variable(_int, "t0")])
        ])
        self.assertSetEqual(set(cond_merge.predecessors), {cond_rhs, cond_short})
        self.assertEqual(cond_merge.successors, [body_block, exit_block])

        self.assertContentsEqual(exit_block, [])
        self.assertEqual(exit_block.predecessors, [cond_merge])
        self.assertEqual(exit_block.successors, [])

    def test_ternary_expression(self):
        code = """
        int foo(int a, int b) {
            int x = a ? b : 3;
            return x;
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 4)

        entry, true_block, false_block, merge_block = fn.basic_blocks
        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(entry.successors, [true_block, false_block])

        self.assertContentsEqual(true_block, [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [Parameter(_int, "b")])
        ])
        self.assertEqual(true_block.predecessors, [entry])
        self.assertEqual(true_block.successors, [merge_block])

        self.assertContentsEqual(false_block, [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [IntegerConstant(3, _int)])
        ])
        self.assertEqual(false_block.predecessors, [entry])
        self.assertEqual(false_block.successors, [merge_block])

        self.assertContentsEqual(merge_block, [
            VarInstruction(COPY_OP, Variable(_int, "x"), [Variable(_unk, "t0")]),
            VarInstruction(RETURN_OP, None, [Variable(_int, "x")])
        ])
        self.assertCountEqual(merge_block.predecessors, [true_block, false_block])
        self.assertEqual(merge_block.successors, [])
    

    def test_nested_ternary_in_if_condition(self):
        code = """
        int foo(int a, int x) {
            if (a ? x + 1 : x * 2) {
                return 3;
            }
            return 4;
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 6)

        entry, cons_block, alt_block, merge_block, then_block, else_block = fn.basic_blocks

        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(entry.predecessors, [])
        self.assertEqual(entry.successors, [cons_block, alt_block])

        self.assertContentsEqual(cons_block, [
            VarInstruction(Addition(), Variable(_unk, "t0"), [Parameter(_int, "x"), IntegerConstant(1, _int)])
        ])
        self.assertEqual(cons_block.predecessors, [entry])
        self.assertEqual(cons_block.successors, [merge_block])

        self.assertContentsEqual(alt_block, [
            VarInstruction(Multiplication(), Variable(_unk, "t0"), [Parameter(_int, "x"), IntegerConstant(2, _int)])
        ])
        self.assertEqual(alt_block.predecessors, [entry])
        self.assertEqual(alt_block.successors, [merge_block])

        self.assertContentsEqual(merge_block, [
            VarInstruction(IF_OP, None, [Variable(_unk, "t0")])
        ])
        self.assertSetEqual(set(merge_block.predecessors), {cons_block, alt_block})
        self.assertEqual(merge_block.successors, [then_block, else_block])

        self.assertContentsEqual(then_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(3, _int)])
        ])
        self.assertEqual(then_block.predecessors, [merge_block])
        self.assertEqual(then_block.successors, [])

        self.assertContentsEqual(else_block, [
            VarInstruction(RETURN_OP, None, [IntegerConstant(4, _int)])
        ])
        self.assertEqual(else_block.predecessors, [merge_block])
        self.assertEqual(else_block.successors, [])

    def test_conditional_expression_in_assignment(self):
        code = """
        int foo(int a, int b) {
            int x;
            x = a ? 5 : b;
            return x;
        }
        """

        fn = self.parse(code)
        self.assertEqual(len(fn.basic_blocks), 4)

        entry, cons_block, alt_block, merge_block = fn.basic_blocks
        self.assertContentsEqual(entry, [
            VarInstruction(IF_OP, None, [Parameter(_int, "a")])
        ])
        self.assertEqual(entry.successors, [cons_block, alt_block])

        self.assertContentsEqual(cons_block, [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [IntegerConstant(5, _int)])
        ])
        self.assertEqual(cons_block.predecessors, [entry])
        self.assertEqual(cons_block.successors, [merge_block])

        self.assertContentsEqual(alt_block, [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [Parameter(_int, "b")])
        ])
        self.assertEqual(alt_block.predecessors, [entry])
        self.assertEqual(alt_block.successors, [merge_block])

        self.assertContentsEqual(merge_block, [
            VarInstruction(COPY_OP, Variable(_int, "x"), [Variable(_unk, "t0")]),
            VarInstruction(RETURN_OP, None, [Variable(_int, "x")])
        ])
        self.assertCountEqual(merge_block.predecessors, [cons_block, alt_block])
        self.assertEqual(merge_block.successors, [])

    def test_postfix_update(self):
        code = """
        void foo(int x, int * src, int * dst) {
            int a = x-- * 2;
            *(dst++) = *(src++);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [Parameter(_int, "x")]),
            VarInstruction(Subtraction(), Parameter(_int, "x"), [Parameter(_int, "x"), IntegerConstant(1, _int)]),
            VarInstruction(Multiplication(), Variable(_int, "a"), [Variable(_unk, "t0"), IntegerConstant(2, _int)]),
            VarInstruction(COPY_OP, Variable(_unk, "t1"), [Parameter(_intp, "dst")]),
            VarInstruction(Addition(), Parameter(_intp, "dst"), [Parameter(_intp, 'dst'), IntegerConstant(1, _int)]),
            VarInstruction(Dereference(), Variable(_unk, "t2"), [Variable(_unk, "t1")]),
            VarInstruction(COPY_OP, Variable(_unk, "t3"), [Parameter(_intp, "src")]),
            VarInstruction(Addition(), Parameter(_intp, "src"), [Parameter(_intp, "src"), IntegerConstant(1, _int)]),
            VarInstruction(Dereference(), Variable(_unk, "t4"), [Variable(_unk, "t3")]),
            VarInstruction(STORE_OP, Variable(_unk, "t2"), [Variable(_unk, "t2"), Variable(_unk, "t4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_function_declarator(self):
        code = """
        void foo(int (*fnptr)(int, int)) {
            int myfndeclarator(int a, float c);
            fnptr(1, 2);
        }
        """

        fn = self.parse(code)

        fnptr_t = Pointer(FunctionType(_int, [(_int, None), (_int, None)]))

        # fnptr being of type Parameter (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the scope.
        correct = [
            VarInstruction(FunctionCall(Parameter(fnptr_t, "fnptr")), Variable(_unk, "t0"), [IntegerConstant(1, _int), IntegerConstant(2, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_pointer_declarators(self):
        code = """
        void foo(int ** x) {
            ++x;
        }
        """

        fn = self.parse(code)

        # x being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            VarInstruction(Addition(), Parameter(Pointer(_intp), "x"), [Parameter(Pointer(_intp), "x"), IntegerConstant(1, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_array_declarators(self):
        code = """
        void foo() {
            int arr[4][5];
            func(arr);
        }
        """

        fn = self.parse(code)

        arr_t = Array(Array(_int, 5), 4)

        # arr being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            VarInstruction(FunctionCall("func"), Variable(_unk, "t0"), [Variable(arr_t, "arr")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_declarator_precedence(self):
        # contains: init_declarator, pointer_declarator, array_declarator, function_declarator, and parenthesized_declarator
        code = """
        void foo() {
            int (*x[8])(int, int) = 0;
        }
        """

        fn = self.parse(code)

        xtype = Array(Pointer(FunctionType(_int, [(_int, None), (_int, None)])), 8)

        # x being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            VarInstruction(COPY_OP, Variable(xtype, "x"), [IntegerConstant(0, _int)]),
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_const_qualifier(self):
        code = """
        void foo() {
            char const *a;
            const char b = 'b';
            const char * const c;
            bar(a, b, c);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_char, "b"), [CharLiteral(ord('b'))]),
            VarInstruction(FunctionCall("bar"), Variable(_unk, "t0"), [Variable(_charp, "a"), Variable(_char, "b"), Variable(_charp, "c")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_return(self):
        code = """
        int foo() {
            return 0;
        }
        """

        fn = self.parse(code)

        correct= [
            VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_compound_expression(self):
        code = """
        void foo(int x, struct values * vals) {
            return vals->init + adjust(x * vals->rate, ADJUSTMENT_FACTOR);
        }
        """

        fn = self.parse(code)

        vals_t = Pointer(IncompleteStruct("values"))

        correct = [
            VarInstruction(MemberAccess(True), Variable(_unk, "t0"), [Parameter(vals_t, "vals"), Field("init")]),
            VarInstruction(MemberAccess(True), Variable(_unk, "t1"), [Parameter(vals_t, "vals"), Field("rate")]),
            VarInstruction(Multiplication(), Variable(_unk, "t2"), [Parameter(_int, "x"), Variable(_unk, "t1")]),
            VarInstruction(FunctionCall("adjust"), Variable(_unk, "t3"), [Variable(_unk, "t2"), GlobalVariable(_unk, "ADJUSTMENT_FACTOR")]),
            VarInstruction(Addition(), Variable(_unk, "t4"), [Variable(_unk, "t0"), Variable(_unk, "t3")]),
            VarInstruction(RETURN_OP, None, [Variable(_unk, "t4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_expression_lval_constant_rval(self):
        code = """
        void foo(int * arr) {
            arr[0] = 1;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(SUBSCRIPT_OP, Variable(_unk, "t0"), [Parameter(_intp, "arr"), IntegerConstant(0, _int)]),
            VarInstruction(STORE_OP, Variable(_unk, "t0"), [Variable(_unk, "t0"), IntegerConstant(1, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_expression_lval_expression_rval(self):
        code = """
        void foo(struct point * pt, int x) {
            pt->x = x + 1;
        }
        """

        pt_t = Pointer(IncompleteStruct("point"))

        fn = self.parse(code)

        correct = [
            VarInstruction(MemberAccess(True), Variable(_unk, "t0"), [Parameter(pt_t, "pt"), Field("x")]),
            VarInstruction(Addition(), Variable(_unk, "t1"), [Parameter(_int, "x"), IntegerConstant(1, _int)]),
            VarInstruction(STORE_OP, Variable(_unk, "t0"), [Variable(_unk, "t0"), Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_expression_lvals(self):
        code = """
        struct point { int x; int y; };
        void foo(struct point * pt1, struct point * pt2, int x) {
            func(pt1, FLAG1)->x = func(pt2, FLAG2)->y = x * 2;
        }
        """

        pt_t = Pointer(Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")]))

        fn = self.parse(code)

        correct  = [
            VarInstruction(FunctionCall("func"), Variable(_unk, "t0"), [Parameter(pt_t, "pt1"), GlobalVariable(_unk, "FLAG1")]),
            VarInstruction(MemberAccess(True), Variable(_unk, "t1"), [Variable(_unk, "t0"), Field("x")]),
            VarInstruction(FunctionCall("func"), Variable(_unk, "t2"), [Parameter(pt_t, "pt2"), GlobalVariable(_unk, "FLAG2")]),
            VarInstruction(MemberAccess(True), Variable(_unk, "t3"), [Variable(_unk, "t2"), Field("y")]),
            VarInstruction(Multiplication(), Variable(_unk, "t4"), [Parameter(_int, "x"), IntegerConstant(2, _int)]),
            VarInstruction(STORE_OP, Variable(_unk, "t3"), [Variable(_unk, "t3"), Variable(_unk, "t4")]),
            VarInstruction(STORE_OP, Variable(_unk, "t1"), [Variable(_unk, "t1"), Variable(_unk, "t3")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_expression_lval_compound_assignment(self):
        code = """
        void foo(int * arr) {
            arr[0] += 1;
            arr[1] *= arr[0] - 2;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(SUBSCRIPT_OP, Variable(_unk, "t0"), [Parameter(_intp, "arr"), IntegerConstant(0, _int)]),
            VarInstruction(Addition(), Variable(_unk, "t1"), [Variable(_unk, "t0"), IntegerConstant(1, _int)]),
            VarInstruction(STORE_OP, Variable(_unk, "t0"), [Variable(_unk, "t0"), Variable(_unk, "t1")]),
            VarInstruction(SUBSCRIPT_OP, Variable(_unk, "t2"), [Parameter(_intp, "arr"), IntegerConstant(1, _int)]),
            VarInstruction(SUBSCRIPT_OP, Variable(_unk, "t3"), [Parameter(_intp, "arr"), IntegerConstant(0, _int)]),
            VarInstruction(Subtraction(), Variable(_unk, "t4"), [Variable(_unk, "t3"), IntegerConstant(2, _int)]),
            VarInstruction(Multiplication(), Variable(_unk, "t5"), [Variable(_unk, "t2"), Variable(_unk, "t4")]),
            VarInstruction(STORE_OP, Variable(_unk, "t2"), [Variable(_unk, "t2"), Variable(_unk, "t5")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_switch(self):
        code = """
        int foo(int c) {
            switch (c) {
                int x = 0;
                case 1: 
                case 2: printf("2"); break;
                case (3): printf("3");
                default:
                    printf("default");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        entry_block = [
            VarInstruction(EqualTo(), Variable(_unk, "t0"), [Parameter(_int, "c"), IntegerConstant(1, _int)]),
            VarInstruction(IF_OP, None, [Variable(_unk, "t0")])
        ]
        case2condition = [
            VarInstruction(EqualTo(), Variable(_unk, "t1"), [Parameter(_int, "c"), IntegerConstant(2, _int)]),
            VarInstruction(IF_OP, None, [Variable(_unk, "t1")])
        ]
        case2body = [VarInstruction(FunctionCall("printf"), Variable(_unk, "t2"), [strlit("2")])]
        case3condition = [
            VarInstruction(EqualTo(), Variable(_unk, "t3"), [Parameter(_int, "c"), IntegerConstant(3, _int)]),
            VarInstruction(IF_OP, None, [Variable(_unk, "t3")])
        ]
        case3body = [VarInstruction(FunctionCall("printf"), Variable(_unk, "t4"), [strlit("3")])]
        default_block = [VarInstruction(FunctionCall("printf"), Variable(_unk, "t5"), [strlit("default")])]
        exit_block = [VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])]

        self.assertContentsEqual(fn.entry_block, entry_block)
        self.assertContentsEqual(fn.basic_blocks[1], case2condition)
        self.assertContentsEqual(fn.basic_blocks[2], case2body)
        self.assertContentsEqual(fn.basic_blocks[3], case3condition)
        self.assertContentsEqual(fn.basic_blocks[4], case3body)
        self.assertContentsEqual(fn.basic_blocks[5], default_block)
        self.assertContentsEqual(fn.basic_blocks[6], exit_block)

    def test_nested_compound_statements(self):
        code = """
        int main() {
            int x = 0;
            {
                int x = 1;
                printf("%d\\n", x);
            }
            printf("%d\\n", x);
        }
        """

        fn = self.parse(code)

        block0 = [
            VarInstruction(COPY_OP, Variable(_int, "x"), [IntegerConstant(0, _int)])
        ]
        block1 = [
            VarInstruction(COPY_OP, Variable(_int, "x"), [IntegerConstant(1, _int)]),
            VarInstruction(FunctionCall("printf"), Variable(_unk, "t0"), [strlit("%d\\n"), Variable(_int, "x")])
        ]
        block2 = [
            VarInstruction(FunctionCall("printf"), Variable(_unk, "t0"), [strlit("%d\\n"), Variable(_int, "x")])
        ]

        self.assertContentsEqual(fn.entry_block, block0)
        self.assertContentsEqual(fn.basic_blocks[1], block1)
        self.assertContentsEqual(fn.basic_blocks[2], block2)
        self.assertEqual(fn.entry_block.instructions[0].result, fn.basic_blocks[2].instructions[0].operands[1], f"Outer scope 'x' is a different variable.")
        self.assertEqual(fn.basic_blocks[1].instructions[0].result, fn.basic_blocks[1].instructions[1].operands[1], f"Inner scope 'x' is a different variable.")
        self.assertNotEqual(fn.entry_block.instructions[0].result, fn.basic_blocks[1].instructions[0].result, f"Variables of the same name in different scopes should be distinct variable objects.")
    
    def test_comma_right_assignment(self):
        code = """
        int main() {
            int a, b, c;
            c = (myfunc(a=4), myfunc(b=5));
            return 0;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_int, "a"), [IntegerConstant(4, _int)]),
            VarInstruction(FunctionCall("myfunc"), Variable(_unk, "t0"), [Variable(_int, "a")]),
            VarInstruction(COPY_OP, Variable(_int, "b"), [IntegerConstant(5, _int)]),
            VarInstruction(FunctionCall("myfunc"), Variable(_int, "c"), [Variable(_int, "b")]),
            VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_for_comma_initialization(self):
        code = """
        void myfn(int n) {
            int i, j;
            for (i=0,j=1;;);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(COPY_OP, Variable(_int, "i"), [IntegerConstant(0, _int)]),
            VarInstruction(COPY_OP, Variable(_int, "j"), [IntegerConstant(1, _int)]),
            VarInstruction(COPY_OP, Variable(_unk, "t0"), [Variable(_int, "j")]), # Unnecessarily (but harmlessly) created by bind_expression
        ]

        self.assertContentsEqual(fn.entry_block, correct)
        self.assertContentsEqual(fn.basic_blocks[1], [VarInstruction(LOOP_OP, None, [IntegerConstant(1, _int)])])

    def test_concatenated_string(self):
        code = """
        void myfn() {
            printf("One" "Two" "Three");
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall("printf"), Variable(_unk, "t0"), [strlit("OneTwoThree")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_compound_literal_expression(self):
        code = """
        void myfn() {
           fn((struct thing){.this = 1 + 7, .that = (44) });
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(Addition(), Variable(_unk, "t0"), [IntegerConstant(1, _int), IntegerConstant(7, _int)]),
            VarInstruction(Initializer(IncompleteStruct("thing"), ["this", "that"]), Variable(_unk, "t1"), [Variable(_unk, "t0"), IntegerConstant(44, _int)]),
            VarInstruction(FunctionCall("fn"), Variable(_unk, "t2"), [Variable(_unk, "t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_ignorable_struct_specifier_and_empty_statement(self):
        code = """
        int main() {
            struct point {
                int x;
                int y;
            };
            ;
            return 0;
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(RETURN_OP, None, [IntegerConstant(0, _int)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_parse_numeric_constants(self):
        code = """
        int main() {
            sumints(1, 2l, 3ll, 4u, 5ul, 6llu);
            sumfloats(1.2, 3.4f, 5.6l);
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(FunctionCall("sumints"), Variable(_unk, "t0"), [
                IntegerConstant(1, _int), IntegerConstant(2, SignedInteger("long", 8)), 
                IntegerConstant(3, SignedInteger("long long", 8)), IntegerConstant(4, UnsignedInteger("unsigned int", 4)), 
                IntegerConstant(5, UnsignedInteger("unsigned long", 8)), IntegerConstant(6, UnsignedInteger("unsigned long long", 8))
            ]),
            VarInstruction(FunctionCall("sumfloats"), Variable(_unk, "t1"), [
                FloatConstant(1.2, Float("double", 8)), FloatConstant(3.4, Float("float", 4)), FloatConstant(5.6, Float("long double", 16))
            ])    
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_char_literal(self):
        code = """
        int iswhitespace(char c) {
            return c == ' ' | c == '\\n';
        }
        """

        fn = self.parse(code)

        correct = [
            VarInstruction(EqualTo(), Variable(_unk, "t0"), [Parameter(_char, 'c'), CharLiteral(ord(' '))]),
            VarInstruction(EqualTo(), Variable(_unk, "t1"), [Parameter(_char, 'c'), CharLiteral(10)]),
            VarInstruction(BitwiseOr(), Variable(_unk, "t2"), [Variable(_unk, "t0"), Variable(_unk, "t1")]),
            VarInstruction(RETURN_OP, None, [Variable(_unk, "t2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    ### Tests some parser components directly
    def test_integer_parsing(self):
        self.assertEqual(parse_int("0xFFFFFFFFLL"), 4294967295)
        self.assertEqual(parse_int("037777777777"), 4294967295)

        self.assertEqual(parse_integer_literal("0x14uLL"), IntegerConstant(20, UnsignedInteger("unsigned long long", 8)))
        self.assertEqual(parse_integer_literal("0xFFFFFFFFLL"), IntegerConstant(-1, SignedInteger("long long", 8)))
        self.assertEqual(parse_integer_literal("0xFFFFFFFELL"), IntegerConstant(-2, SignedInteger("long long", 8)))
        self.assertEqual(parse_integer_literal("0xFFFFFFFFFFFFFC00LL"), IntegerConstant(-1024, SignedInteger("long long", 8)))
        self.assertEqual(parse_integer_literal("037777777777"), IntegerConstant(4294967295, UnsignedInteger("unsigned int", 4)))
        self.assertEqual(parse_integer_literal("037777777776"), IntegerConstant(4294967294, UnsignedInteger("unsigned int", 4)))
        self.assertEqual(parse_integer_literal("00"), IntegerConstant(0, SignedInteger("int", 4)))

        self.assertEqual(parse_integer_literal("0xA"), IntegerConstant(10, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("0xB"), IntegerConstant(11, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("0xA0"), IntegerConstant(160, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("02"), IntegerConstant(2, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("0600"), IntegerConstant(384, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("-1"), IntegerConstant(-1, SignedInteger("int", 4)))
        self.assertEqual(integer_literal_components("10"), (10, 10, "", None))

        self.assertEqual(parse_integer_literal("2147483647"), IntegerConstant(2 ** 31 - 1, SignedInteger("int", 4)))
        self.assertEqual(parse_integer_literal("2147483648"), IntegerConstant(2 ** 31, SignedInteger("long", 8)))

# Ensures the CFG is correct. Does not check the contents of the basic blocks.
# These tests are sensitive to the order that the basic blocks are stored in a Function's basic_block list.
class TestCFG(unittest.TestCase):
    def parse(self, code: str):
        return compile(bytes(code, "utf8"))[0]
    
    def test_if(self):
        code = """
        int foo(int x) {
            if (x) {
                print("x is positive.");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]
        
        assert len(exit_block.predecessors) == 2
        assert entry_block in exit_block.predecessors
        assert if_body in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_terminating_if(self):
        code = """
        void foo(int x) {
            if (x) {
                print("x is positive.");
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 2
        assert entry_block in exit_block.predecessors
        assert if_body in exit_block.predecessors
        assert len(exit_block.successors) == 0
    
    def test_if_else(self):
        code = """
        int foo(int x) {
            if (x)
                print("x is positive.");
            else
                print("x is not positive.");
            
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        else_block = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert else_block in entry_block.successors
        
        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]

        assert else_block.predecessors == [entry_block]
        assert else_block.successors == [exit_block]

        assert len(exit_block.predecessors) == 2
        assert if_body in exit_block.predecessors
        assert else_block in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_terminating_if_else(self):
        code = """
        void foo(int x) {
            if (x) {
                print("x is positive.");
            } else {
                print("x is not positive.");
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        else_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert else_block in entry_block.successors
        
        assert if_body.predecessors == [entry_block]
        assert if_body.successors == []

        assert else_block.predecessors == [entry_block]
        assert else_block.successors == []

    def test_else_if(self):
        code = """
        int foo(int a, int b) {
            if (a > b) {
                printf("A")
            } else if (a < b) {
                print("B")
            }
            return a < b
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        true_block = fn.basic_blocks[1]
        false_block = fn.basic_blocks[2]
        false_true_block = fn.basic_blocks[3]
        return_block = fn.basic_blocks[4]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) ==  2
        assert true_block in entry_block.successors
        assert false_block in entry_block.successors

        assert true_block.predecessors == [entry_block]
        assert true_block.successors == [return_block]

        assert false_block.predecessors == [entry_block]
        assert len(false_block.successors) == 2
        assert false_true_block in false_block.successors
        assert return_block in false_block.successors
        
        assert false_true_block.predecessors == [false_block]
        assert false_true_block.successors == [return_block]

        assert len(return_block.predecessors) == 3
        assert true_block in return_block.predecessors
        assert false_block in return_block.predecessors
        assert false_true_block in return_block.predecessors
        assert len(return_block.successors) == 0
    
    def test_else_if_while_else(self):
        code = """
        int foo(int a, int b) {
            if (a < b) {
                print("A");
            } else if (a > b) {
                while (flag) doit();
            } else {
                print("Same");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 7
        entry_block = fn.entry_block
        true_block = fn.basic_blocks[1]
        middle_block = fn.basic_blocks[2]
        while_condition = fn.basic_blocks[3]
        while_body = fn.basic_blocks[4]
        else_block = fn.basic_blocks[5]
        return_block = fn.basic_blocks[6]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert true_block in entry_block.successors
        assert middle_block in entry_block.successors

        assert true_block.predecessors == [entry_block]
        assert true_block.successors == [return_block]

        assert middle_block.predecessors == [entry_block]
        assert len(middle_block.successors) == 2
        assert while_condition in middle_block.successors
        assert else_block in middle_block.successors

        assert len(while_condition.predecessors) == 2
        assert middle_block in while_condition.predecessors
        assert while_body in while_condition.predecessors
        assert len(while_condition.successors) == 2
        assert return_block in while_condition.successors
        assert while_body in while_condition.successors

        assert while_body.predecessors == [while_condition]
        assert while_body.successors == [while_condition]

        assert else_block.predecessors == [middle_block]
        assert else_block.successors == [return_block]

        assert len(return_block.predecessors) == 3
        assert true_block in return_block.predecessors
        assert while_condition in return_block.predecessors
        assert else_block in return_block.predecessors
        assert len(return_block.successors) == 0
    
    def test_for(self):
        code  = """
        int foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_update]

        assert loop_update.predecessors == [loop_body]
        assert loop_update.successors == [loop_test]

        assert exit_block.predecessors== [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_for(self):
        code  = """
        void foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_update]

        assert loop_update.predecessors == [loop_body]
        assert loop_update.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_while(self):
        code = """
        int foo(char * message) {
            while (rand() > 0.4) {
                printf(message);
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_body in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_while(self):
        code = """
        void foo(char * message) {
            while (rand() > 0.4) {
                printf(message);
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_body in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_do_while(self):
        code = """
        int foo(char * message) {
            do {
                printf(message);
            } while (rand() > 0.4);
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_body]
        
        assert len(loop_body.predecessors) == 2
        assert entry_block in loop_body.predecessors
        assert loop_test in loop_body.predecessors
        assert loop_body.successors == [loop_test]

        assert loop_test.predecessors == [loop_body]
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert exit_block.predecessors == [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_do_while(self):
        code = """
        int foo(char * message) {
            do {
                printf(message);
            } while (rand() > 0.4);
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_body]
        
        assert len(loop_body.predecessors) == 2
        assert entry_block in loop_body.predecessors
        assert loop_test in loop_body.predecessors
        assert loop_body.successors == [loop_test]

        assert loop_test.predecessors == [loop_body]
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_if_return(self):
        code = """
        int foo(int x) {
            if (x) {
                return 1;
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == []
        
        assert exit_block.predecessors == [entry_block]
        assert exit_block.successors == []
    
    def test_for_if_return(self):
        code = """
        int foo(int x) {
            int i;
            bar();
            for (int i = 0; i < gbl; i++) {
                if (gbl2 == x){
                    return 1;
                }
            } 
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 6
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body_if = fn.basic_blocks[3]
        if_body = fn.basic_blocks[4]
        post_loop = fn.basic_blocks[5]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]
        
        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body_if in loop_test.successors
        assert post_loop in loop_test.successors

        assert loop_body_if.predecessors == [loop_test]
        assert len(loop_body_if.successors) == 2
        assert if_body in loop_body_if.successors
        assert loop_update in loop_body_if.successors

        assert if_body.predecessors == [loop_body_if]
        assert if_body.successors == [] # has a return statement.

        assert loop_update.predecessors == [loop_body_if]
        assert loop_update.successors == [loop_test]

        assert post_loop.predecessors == [loop_test]
        assert post_loop.successors == []
    
    def test_if_if(self):
        code = """
        int foo(int x, int y) {
            if (x) {
                if (y) {
                    print("done.");
                }
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        inner_if = fn.basic_blocks[1]
        inner_if_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert inner_if in entry_block.successors
        assert exit_block in entry_block.successors
        
        assert inner_if.predecessors == [entry_block]
        assert len(inner_if.successors) == 2
        assert inner_if_body in inner_if.successors
        assert exit_block in inner_if.successors

        assert inner_if_body.predecessors == [inner_if]
        assert inner_if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 3
        assert entry_block in exit_block.predecessors
        assert inner_if in exit_block.predecessors
        assert inner_if_body in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_if_if_if(self):
        code = """
        int foo(int x) {
            if (x > 0) {
                if (x < 30) {
                    if (y) {
                        print("passed.");
                    }
                }
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        middle_if = fn.basic_blocks[1]
        inner_if = fn.basic_blocks[2]
        inner_if_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert middle_if in entry_block.successors
        assert exit_block in entry_block.successors

        assert middle_if.predecessors == [entry_block]
        assert len(middle_if.successors) == 2
        assert inner_if in middle_if.successors
        assert exit_block in middle_if.successors

        assert inner_if.predecessors == [middle_if]
        assert len(inner_if.successors) == 2
        assert inner_if_body in inner_if.successors
        assert exit_block in inner_if.successors

        assert inner_if_body.predecessors == [inner_if]
        assert inner_if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 4
        assert entry_block in exit_block.predecessors
        assert middle_if in exit_block.predecessors
        assert inner_if in exit_block.predecessors
        assert inner_if_body in exit_block.predecessors
        assert exit_block.successors == []

    def test_switch_return(self):
        code = """
        int compute_size(int type) {
            switch (type) {
            case 1:
                return 4;
            case 2:
                return 8;
            default:
                return 12;
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5

        entry_block = fn.basic_blocks[0]
        case1_block = fn.basic_blocks[1]
        test2_block = fn.basic_blocks[2]
        case2_block = fn.basic_blocks[3]
        default_block = fn.basic_blocks[4]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert case1_block in entry_block.successors
        assert test2_block in entry_block.successors

        assert len(case1_block.predecessors) == 1
        assert entry_block in case1_block.predecessors
        assert len(case1_block.successors) == 0

        assert len(test2_block.predecessors) == 1
        assert entry_block in test2_block.predecessors
        assert len(test2_block.successors) == 2
        assert case2_block in test2_block.successors
        assert default_block in test2_block.successors
        
        assert len(case2_block.predecessors) == 1
        assert test2_block in case2_block.predecessors
        assert len(case2_block.successors) == 0

        assert len(default_block.predecessors) == 1
        assert test2_block in default_block.predecessors
        assert len(default_block.successors) == 0
    
    def test_unreachable_block(self):
        code = """
        int bar(int x, int y) {
            if (x) {
                return -y;
            } else {
                return y;
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3

        entry_block = fn.basic_blocks[0]
        true_block = fn.basic_blocks[1]
        false_block = fn.basic_blocks[2]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert true_block in entry_block.successors
        assert false_block in entry_block.successors

        assert len(true_block.predecessors) == 1
        assert entry_block in true_block.predecessors
        assert len(true_block.successors) == 0

        assert len(false_block.predecessors) == 1
        assert entry_block in false_block.predecessors
        assert len(false_block.successors) == 0
    
    def test_return_in_nested_compound_statement(self):
        code = """
        int fn(int x) {
            x = x + 4;
            {
                x = x * 7;
                return x;
            }
            x = x + 8;
            return x;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 2
        
        entry_block = fn.basic_blocks[0]
        return_block = fn.basic_blocks[1]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 1
        assert return_block in entry_block.successors

        assert len(return_block.predecessors) == 1
        assert entry_block in return_block.predecessors
        assert len(return_block.successors) == 0




if __name__ == '__main__':
    unittest.main()
