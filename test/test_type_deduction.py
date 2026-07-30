"""Test type inference rules individually or in combination.
"""

import unittest

from faultless.ir import *
from faultless.c import compile, PRIMITIVE_TYPES
from faultless.analysis import deduce_types
from .utils import strlit

_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]
_char = PRIMITIVE_TYPES["char"]
_int = PRIMITIVE_TYPES["int"]
_long = PRIMITIVE_TYPES["long"]
_unk = UnknownType()


class TestOperationTypeInferenceRules(unittest.TestCase):
    def assertTypeInferenceFails(self, operation: Operation, *arguments: Constant | Variable):
        try:
            retval = operation.deduce_type(*arguments)
            assert False, f"{operation.sprint(*arguments)} should fail with types " + ', '.join(str(typeof(a)) for a in arguments) + f" but got {retval}"
        except SemanticError:
            pass

    def test_arithmetic_type_conversion(self):
        # Floating point numbers
        self.assertEqual(PRIMITIVE_TYPES["long double"], arithmetic_type_conversion(PRIMITIVE_TYPES["long double"], _int))
        self.assertEqual(PRIMITIVE_TYPES["double"], arithmetic_type_conversion(PRIMITIVE_TYPES["double"], _float))
        # Integers
        self.assertEqual(_int, arithmetic_type_conversion(_int, _int))
        self.assertEqual(_long, arithmetic_type_conversion(_long, _int))
        self.assertEqual(_int, arithmetic_type_conversion(_char, PRIMITIVE_TYPES["unsigned short"])) # due to integer promotion
        self.assertEqual(PRIMITIVE_TYPES["unsigned long"], arithmetic_type_conversion(PRIMITIVE_TYPES["unsigned long"], PRIMITIVE_TYPES["unsigned short"]))
        self.assertEqual(PRIMITIVE_TYPES["unsigned int"], arithmetic_type_conversion(PRIMITIVE_TYPES["unsigned int"], _int))
        self.assertEqual(PRIMITIVE_TYPES["unsigned long long"], arithmetic_type_conversion(PRIMITIVE_TYPES["unsigned long long"], _int))
        self.assertEqual(_int, arithmetic_type_conversion(_int, PRIMITIVE_TYPES["unsigned short"]))

    def test_addition(self):
        addition = Addition()
        # Mostly a test of arithmetic_type_conversion
        res = addition.deduce_type(Variable(_int, "a"), IntegerConstant(1, _int))
        self.assertEqual(res, _int)
        res = addition.deduce_type(Variable(_long, "a"), IntegerConstant(1, _int))
        self.assertEqual(res, _long)
        res = addition.deduce_type(Variable(_float, "a"), Variable(_long, "b"))
        self.assertEqual(res, _float)
        res = addition.deduce_type(Variable(PRIMITIVE_TYPES["long double"], "a"), Variable(_int, "b"))
        self.assertEqual(res, PRIMITIVE_TYPES["long double"])
        res = addition.deduce_type(Variable(PRIMITIVE_TYPES["unsigned int"], "a"), Variable(_long, "b"))
        self.assertEqual(res, _long)
        
        self.assertEqual(_unk, addition.deduce_type(Variable(_unk, "t0"), Variable(_int, "a")))
        
        # Testing pointer arithmetic
        res = addition.deduce_type(Variable(Pointer(_int), "a"), IntegerConstant(1, _int))
        self.assertEqual(res, Pointer(_int))
        res = addition.deduce_type(Variable(_long, "a"), Variable(Pointer(_float), "b"))
        self.assertEqual(res, Pointer(_float))

        # Check error handling
        self.assertTypeInferenceFails(addition, Variable(Struct("s", [UDT.Field(_int, 'a')]), 'b'), Variable(_int, 'c'))
        self.assertTypeInferenceFails(addition, Variable(Pointer(_int), "a"), Variable(Pointer(_int), "b"))


    def test_subtraction(self):
        subtraction = Subtraction()
        res = subtraction.deduce_type(Variable(Pointer(_float), "a"), IntegerConstant(1, _int))
        self.assertEqual(res, Pointer(_float))
        res = subtraction.deduce_type(Variable(Array(_int, 5), "a"), Variable(Array(_int, 5), "b"))
        self.assertEqual(res, SIZE_T)
        res = subtraction.deduce_type(Variable(_int, "a"), Variable(PRIMITIVE_TYPES["unsigned char"], "b"))
        self.assertEqual(_int, res)
        self.assertEqual(_unk, subtraction.deduce_type(Variable(_unk, "t0"), Variable(_int, "a")))

        self.assertTypeInferenceFails(subtraction, Variable(Pointer(_int), "a"), Variable(Pointer(_float), "b"))

    def test_multiplicative_ops(self):
        self.assertTypeInferenceFails(Multiplication(), IntegerConstant(3, _int), Variable(Pointer(Void()), "b"))

        self.assertEqual(_float, Division().deduce_type(Variable(_float, "a"), Variable(_int, "b")))
        self.assertEqual(_long, ModulusDivision().deduce_type(Variable(_int, "a"), Variable(_long, "b")))
        self.assertEqual(_unk, Multiplication().deduce_type(Variable(_unk, "t0"), Variable(_int, "a")))

    def test_shift_ops(self):
        self.assertTypeInferenceFails(LeftShift(), Variable(_int, "a"), FloatConstant(2.2, _float))
        self.assertEqual(_int, RightShift().deduce_type(Variable(_int, "a"), IntegerConstant(2, _int)))
        self.assertEqual(_unk, RightShift().deduce_type(Variable(_unk, "a"), IntegerConstant(2, _int)))

    def test_relational_ops(self):
        self.assertTypeInferenceFails(LessThanOrEqualTo(), Variable(Pointer(_int), "a"), Variable(Pointer(_float), "b"))
        self.assertTypeInferenceFails(GreaterThan(), Variable(_int, "a"), Variable(Array(_int, 5), "b"))

        self.assertEqual(_int, LessThan().deduce_type(Variable(Pointer(_int), "b"), Variable(Array(_int, 4), "a")))
        self.assertEqual(_int, GreaterThanOrEqualTo().deduce_type(Variable(_int, "b"), Variable(_int, "a")))

        self.assertEqual(_int, LessThan().deduce_type(Variable(_unk, "t0"), Variable(Pointer(_int), "b")))
        self.assertEqual(_int, GreaterThan().deduce_type(Variable(_unk, "t0"), Variable(_float, "a")))

    def test_equality_ops(self):
        equality = EqualTo()
        inequality = NotEqualTo()
        
        self.assertTypeInferenceFails(equality, Variable(Pointer(_int), "a"), Variable(_int, "b"))
        self.assertTypeInferenceFails(equality, Variable(Pointer(_int), "b"), Variable(Pointer(_float), "c"))
        self.assertTypeInferenceFails(inequality, Variable(Pointer(_int), "c"), IntegerConstant(1, _int))

        self.assertEqual(_int, equality.deduce_type(Variable(Pointer(Void()), "a"), Variable(Pointer(_int), "b")))
        self.assertEqual(_int, equality.deduce_type(Variable(Pointer(_int), "a"), IntegerConstant(0, _long)))
        self.assertEqual(_int, equality.deduce_type(Variable(_float, "a"), Variable(_int, "b")))
        
        self.assertEqual(_int, equality.deduce_type(Variable(_unk, "t0"), Variable(_float, "b")))
        self.assertEqual(_int, equality.deduce_type(Variable(Pointer(_int), "a"), Variable(_unk, "t1")))
        self.assertEqual(_int, equality.deduce_type(Variable(Pointer(_int), "a"), Variable(Pointer(_unk), "t1")))

    def test_bitwise_ops(self):
        self.assertTypeInferenceFails(BitwiseAnd(), Variable(Pointer(_int), "a"), Variable(_int, "b"))
        
        self.assertEqual(_long, BitwiseOr().deduce_type(Variable(PRIMITIVE_TYPES["short"], "a"), Variable(_long, "b")))
        self.assertEqual(_long, BitwiseXOr().deduce_type(Variable(_long, "a"), Variable(_int, "b")))
        self.assertEqual(_long, BitwiseNot().deduce_type(Variable(_long, "b")))
        self.assertEqual(_unk, BitwiseOr().deduce_type(Variable(_unk, "t0"), Variable(_int, "a")))
        self.assertEqual(_unk, BitwiseNot().deduce_type(Variable(_unk, "b")))

    def test_logical_ops(self):
        self.assertEqual(_int, LogicalAnd().deduce_type(Variable(_int, "a"), Variable(_long, "b")))
        self.assertTypeInferenceFails(LogicalOr(), Variable(_int, "a"), Variable(IncompleteStruct("thing"), "b"))
        self.assertEqual(_int, LogicalNot().deduce_type(Variable(Pointer(_int), "a")))

        self.assertEqual(_int, LogicalAnd().deduce_type(Variable(_unk, "a"), Variable(_long, "b")))
        self.assertEqual(_int, LogicalNot().deduce_type(Variable(_unk, "a")))

    def test_conditional_op(self):
        conditional = ConditionalOperation()

        pt = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])

        self.assertEqual(Pointer(Void()), conditional.deduce_type(Variable(_int, "a"), Variable(Pointer(_int), "b"), Variable(Pointer(Void()), "c")))
        self.assertEqual(pt, conditional.deduce_type(Variable(_long, "a"), Variable(pt, "b"), Variable(pt, "c")))
        self.assertEqual(_double, conditional.deduce_type(Variable(_int, "a"), Variable(_int, "b"), Variable(_double, "c")))
        self.assertEqual(Pointer(_int), conditional.deduce_type(Variable(_int, "a"), IntegerConstant(0, _long), Variable(Pointer(_int), "c")))
        self.assertTypeInferenceFails(conditional, Variable(pt, "a"), Variable(_int, "b"), Variable(_int, "c"))
        self.assertTypeInferenceFails(conditional, Variable(_int, "a"), Variable(Pointer(_int), "b"), Variable(Pointer(_double), "c"))

        self.assertEqual(_double, conditional.deduce_type(Variable(_unk, "t0"), Variable(_int, "b"), Variable(_double, "c")))
        self.assertEqual(_unk, conditional.deduce_type(Variable(_int, "a"), Variable(_unk, "t0"), Variable(_double, "c")))
        self.assertEqual(_unk, conditional.deduce_type(Variable(_int, "a"), Variable(_int, "b"), Variable(_unk, "t0")))

    def test_cast_op(self):
        cast = Cast()

        pt = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])

        self.assertEqual(Pointer(_long), cast.deduce_type(Pointer(_long), Variable(_long, "v1")))
        self.assertEqual(_float, cast.deduce_type(_float, Variable(_long, "b")))
        self.assertEqual(_float, cast.deduce_type(_float, Variable(_unk, "t0")))
        try:
            cast.deduce_type(pt, Variable(_long, "l"))
            assert False, f"Cannot cast to struct type."
        except SemanticError:
            pass

    def test_sizeof_op(self):
        sizeof = SizeOf()

        self.assertEqual(SIZE_T, sizeof.deduce_type(_int))
        self.assertEqual(SIZE_T, sizeof.deduce_type(Variable(_int, "x")))
        self.assertEqual(SIZE_T, sizeof.deduce_type(Variable(_unk, "t0")))
        self.assertTypeInferenceFails(sizeof, Variable(FunctionType(_int, []), "fn"))
        try:
            sizeof.deduce_type(FunctionType(_int, []))
            assert False, f"Cannot cast to struct type."
        except SemanticError:
            pass

    def test_unary_minus(self):
        minus = UnaryMinus()
        
        self.assertEqual(_int, minus.deduce_type(Variable(PRIMITIVE_TYPES["short"], "a")))
        self.assertEqual(_long, minus.deduce_type(Variable(_long, "a")))
        self.assertEqual(_unk, minus.deduce_type(Variable(_unk, "t0")))
        self.assertTypeInferenceFails(minus, Variable(Pointer(_int), "a"))

    def test_pointer_ops(self):
        addressof = AddressOf()
        dereference = Dereference()

        struct_t = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])

        self.assertEqual(Pointer(_int), addressof.deduce_type(Variable(_int, "a")))
        self.assertEqual(Pointer(struct_t), addressof.deduce_type(Variable(struct_t, "pt")))
        self.assertEqual(Pointer(_unk), addressof.deduce_type(Variable(_unk, "t0")))
        self.assertTypeInferenceFails(addressof, FloatConstant(2.2, _float))

        self.assertEqual(struct_t, dereference.deduce_type(Variable(Pointer(struct_t), "a")))
        self.assertTypeInferenceFails(dereference, _int)

    def test_subscript_op(self):
        subscript = Subscript()

        self.assertEqual(_int, subscript.deduce_type(Variable(Array(_int, 8), "b"), IntegerConstant(4, _int)))
        self.assertEqual(_int, subscript.deduce_type(Variable(Pointer(_int), "d"), Variable(_int, "c")))
        self.assertEqual(_unk, subscript.deduce_type(Variable(_unk, "t0"), IntegerConstant(0, _int)))
        self.assertEqual(_int, subscript.deduce_type(Variable(Array(_int, 8), "a"), Variable(_unk, "t0")))
        self.assertTypeInferenceFails(subscript, Variable(Pointer(_int), "a"), FloatConstant(2.4, _double))
        self.assertTypeInferenceFails(subscript, Variable(IncompleteUnion("u"), "a"), IntegerConstant(2, _int))

    def test_member_access(self):
        dot = MemberAccess(False)
        arrow = MemberAccess(True)

        hash_t = Struct("hash_t", [
            UDT.Field(_long, "nelements"),
            UDT.Field(_long, "size"),
            UDT.Field(Pointer(Pointer(Void())), "table")
        ])

        self.assertEqual(_long, dot.deduce_type(Variable(hash_t, "h"), Field("size")))
        self.assertEqual(Pointer(Pointer(Void())), arrow.deduce_type(Variable(Pointer(hash_t), "h"), Field("table")))
        self.assertEqual(_unk, arrow.deduce_type(Variable(Pointer(IncompleteStruct("thing")), "a"), Field("b")))
        self.assertEqual(_unk, arrow.deduce_type(Variable(_unk, "t0"), Field("address")))
        self.assertTypeInferenceFails(dot, Variable(hash_t, "h"), Field("length"))
        self.assertTypeInferenceFails(dot, Variable(Array(_int, 3), "h"), Field("size"))
        self.assertTypeInferenceFails(arrow, Variable(hash_t, "h"), Field("size"))
        self.assertTypeInferenceFails(arrow, Variable(_unk, "t0"), IntegerConstant(4, _int))


    def test_function_call(self):
        realloc = FunctionCall("realloc", FunctionType(Pointer(Void()), [(Pointer(Void()), "ptr"), (SIZE_T, "size")]))
        printf = FunctionCall("printf", FunctionType(_int, [(Pointer(_char), "format"), (FunctionType.VariadicParameter(), None)]))
        foo = FunctionCall("foo", FunctionType(_int, [(Pointer(_int), "a"), (Array(_int, 19), "b")]))

        self.assertEqual(Pointer(Void()), realloc.deduce_type(Variable(Pointer(_int), "a"), IntegerConstant(20, _int)))
        self.assertEqual(Pointer(Void()), realloc.deduce_type(Variable(Array(_int, 20), "a"), Variable(_float, "b")))
        self.assertEqual(Pointer(Void()), realloc.deduce_type(Variable(_unk, "t0"), Variable(_unk, "t1")))
        self.assertTypeInferenceFails(realloc, Variable(IncompleteStruct("thing"), "a"), Variable(_float, "b"))

        self.assertEqual(_int, printf.deduce_type(strlit("%d\\n"), Variable(_int, "x")))
        self.assertTypeInferenceFails(printf, Variable(_int, "x"), strlit("%d\\n"))

        self.assertTypeInferenceFails(foo, Variable(_int, "i"), Variable(_float, "f"))
        self.assertTypeInferenceFails(foo, Variable(Array(_float, 1), "a"), Variable(_int, "b"))

        self.assertEqual(_unk, FunctionCall("bar").deduce_type(Variable(_int, "a"), Variable(_float, "b")))

    def test_control_flow_ops(self):
        self.assertEqual(None, If().deduce_type(Variable(_unk, "t0")))
        self.assertEqual(None, LoopOp().deduce_type(Variable(_unk, "t0")))
        self.assertEqual(None, Return().deduce_type(Variable(_unk, "t0")))
        self.assertTypeInferenceFails(If(), Variable(IncompleteStruct("thing"), "a"))
        self.assertTypeInferenceFails(LoopOp(), Variable(IncompleteStruct("thing"), "b"))
        self.assertTypeInferenceFails(Return(), Variable(Array(_int, 20), "b"))

    def test_init_op_with_arrays(self):
        self.assertEqual(Array(_int, 4), Initializer(Array(_int, 4)).deduce_type(IntegerConstant(0, _int), IntegerConstant(1, _int), IntegerConstant(2, _int), Variable(_int, "x")))
        self.assertEqual(Array(_float, 2), Initializer(Array(_float, 2)).deduce_type(FloatConstant(3.2, _float), FloatConstant(4.5, _float)))
        self.assertEqual(Array(_int, 2), Initializer(Array(_int, 2)).deduce_type(Variable(_unk, "t0"), IntegerConstant(3, _int)))
        self.assertEqual(Array(Pointer(_int), 3), Initializer(Array(Pointer(_int), 3)).deduce_type(Variable(Pointer(_int), "a"), Variable(Pointer(_int), "b"), IntegerConstant(0, _int)))
        self.assertEqual(Array(Pointer(Void()), 3), Initializer(Array(Pointer(Void()), 3)).deduce_type(Variable(Pointer(_int), "a"), Variable(Pointer(_float), "b"), IntegerConstant(0, _long)))
        self.assertEqual(Array(IncompleteStruct("thing"), 3), Initializer(Array(IncompleteStruct("thing"), 3)).deduce_type(Variable(IncompleteStruct("thing"), "a"), Variable(IncompleteStruct("thing"), "b"), Variable(IncompleteStruct("thing"), "c")))
        
        self.assertTypeInferenceFails(Initializer(Array(Pointer(_int), 3)), Variable(Pointer(_int), "a"), Variable(Pointer(_int), "b"), IntegerConstant(2, _int))
        self.assertTypeInferenceFails(Initializer(Array(_int, 2)), Variable(IncompleteStruct("thing"), "a"), Variable(_int, "b"))
        self.assertTypeInferenceFails(Initializer(Array(Array(_int, 3), 2)), Variable(Array(_int, 3), "b"), Variable(Pointer(_int), "a"))

    def test_init_op_with_structs(self):
        hash_t = Struct("hash_t", [
            UDT.Field(_long, "nelements"),
            UDT.Field(_long, "size"),
            UDT.Field(Pointer(Pointer(Void())), "table")
        ])

        incomplete_type = Initializer(hash_t.stub, ["size", "table"])
        complete_type = Initializer(hash_t, ["nelements", "table"])

        self.assertEqual(hash_t.stub, incomplete_type.deduce_type(IntegerConstant(8, _int), Parameter(Pointer(Void()), "t")))
        self.assertEqual(hash_t, complete_type.deduce_type(IntegerConstant(0, _int), Variable(Pointer(Pointer(Void())), "ptr")))
        self.assertEqual(hash_t, complete_type.deduce_type(Variable(_int, "n"), Variable(Pointer(Pointer(_int)), "b")))
        self.assertEqual(hash_t, complete_type.deduce_type(Variable(_unk, "n"), Variable(_unk, "b")))

        self.assertTypeInferenceFails(complete_type, Variable(Pointer(Pointer(Void())), "ptr"), IntegerConstant(0, _int))
        self.assertTypeInferenceFails(complete_type, Variable(_int, "n"), Variable(Pointer(_int), "b"))

    def test_phi_op(self):
        phi = Phi(Variable(_int, "i")) # The argument here doesn't matter for the type inference code.
        self.assertTypeInferenceFails(phi, Variable(_long, "a"), Variable(_int, "a"))
        self.assertTypeInferenceFails(phi, Variable(Pointer(_int), "a"), Variable(_int, "a"))
        self.assertTypeInferenceFails(phi, FloatConstant(3.3, _double), Variable(_int, 'x'), Variable(_double, "x")) 
        self.assertEqual(_long, phi.deduce_type(IntegerConstant(0, _int), Variable(_long, "a")))
        self.assertEqual(_int, phi.deduce_type(IntegerConstant(0, _int), GlobalVariable(_unk, "a"), GlobalVariable(_unk, "a")))

    def test_copy_op(self):
        self.assertEqual(_double, Copy().deduce_type(FloatConstant(8.0, _double)))

class TestChainedTypeInference(unittest.TestCase):
    def check_inferred_types(self, code: str, types: list[list[CType | None]]):
        ir = compile(bytes(code, "utf8"))[0]
        deduce_types(ir)
        self.assertEqual(len(ir.basic_blocks), len(types))
        for i, (bb, ts) in enumerate(zip(ir, types)):
            self.assertEqual(len(bb.instructions), len(ts), f"Expected IR for block {i}:\n  " + '\n  '.join(str(ins) for ins in bb.instructions))
            for j, (instruction, t) in enumerate(zip(bb, ts)):
                if instruction.result is not None:
                    self.assertEqual(instruction.result.type, t, f"Mismatched types for instruction {instruction} (block {i}, instruction {j})")
                else:
                    self.assertEqual(instruction.result, t, f"Instruction {instruction} should have no return type (block {i}, instruction {j}).")

    def test_long_short_and_char(self):
        code = """
        long math(int x) {
            return x + 4l - 'c';
        }
        """
        oracle = [[_long, _long, None]]
        self.check_inferred_types(code, oracle)

    def test_pointer_arithmetic(self):
        code = """
        void initialize_matrix(int *matrix) {
            for (int i = 0; i < ROWS * COLS; ++i) {
                *(matrix + i) = i + 1;
            }
        }
        """
        oracle = [
            [_int],
            [_unk, _int, None],
            [_int],
            [Pointer(_int), _int, _int, _int],
            []
        ]
        self.check_inferred_types(code, oracle)

    def test_unknown_array_init(self):
        code = """
        void randomize_global_array(long long seed) {
            for (int i = 0; i < LENGTH; ++i) {
                arr[i] = random(seed);
            }
        }
        """
        oracle = [
            [_int],
            [_int, None],
            [_int],
            [_unk, _unk, _unk],
            []
        ]
        self.check_inferred_types(code, oracle)

    def test_union_access(self):
        code = """
        union ctoi { int val; char cs[4]; };
        int foo(char code[3]) {
            union ctoi intify = { .cs={code[0], code[1], code[2], '\\0'} };
            return intify.val;
        }
        """
        oracle = [[
            _char, _char, _char, # array access
            Array(_char, 4), # nested initializer
            Union("ctoi", [UDT.Field(_int, "val"), UDT.Field(Array(_char, 4), "cs")]),
            _int, # member access
            None # return
        ]]
        self.check_inferred_types(code, oracle)

    def test_incomplete_struct_access(self):
        code = """
        int grow(struct array *arr) {
            int element_size = sizeof(arr->array[0]);
            int new_length = arr->len + arr_len_alloc(element_size);
            void **new = realloc(arr->array, new_length * element_size);
            if (new == 0)
                return -1;
            arr->array = new;
            arr->len = new_length;
            return 0;
        }
        """
        oracle = [
            [
                _unk, # arr->array
                _unk, # array[0]
                _int, # sizeof
                _unk, # arr-> len
                _unk, # arr_len_alloc(element_size)
                _int, # +
                _unk, # arr->array,
                _int, # new_length * element_size
                Pointer(Pointer(Void())), # realloc
                _int, # new == 0
                None # if
            ],
            [None], # if body
            [
                _unk, # arr->array 
                _unk, # store
                _unk, # arr->len
                _unk, # store
                None # return
            ]
        ]
        self.check_inferred_types(code, oracle)

    def test_cast_and_defined_struct(self):
        code = """
        struct timeval { long tv_sec; long tv_usec; };
        double get_current_milisec() {
            struct timeval tv;
            gettimeofday(&tv, (void *)0);
            return tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
        }
        """
        timeval = Struct("timeval", [UDT.Field(_long, "tv_sec"), UDT.Field(_long, "tv_usec")])
        oracle = [[
            Pointer(timeval),
            Pointer(Void()), # (void *)0
            _unk,
            _long, # tv.tv_sec,
            _double, # * 1000
            _long, # tv.tv_usec
            _double, # (double)
            _double, # /
            _double, # +
            None # return
        ]]
        self.check_inferred_types(code, oracle)
    
    def test_incomplete_struct_types(self):
        code = """
        unsigned short checksum(unsigned short *address, int length) {
            if (dontchksum(((struct ip *)address)->ip_src.s_addr))
                return 0;
            return check_ext(address, length << 2, 0);
        }
        """
        ip_t = IncompleteStruct("ip")
        oracle = [
            [
                Pointer(ip_t), # (struct ip *)address
                _unk, # ->ip_src
                _unk, # .s_addr
                _unk, # dontchksum(...)
                None # if
            ],
            [None], # if body: return 0
            [
                _int, # length << 2
                _unk, # check_ext(...)
                None # return
            ]
        ]
        self.check_inferred_types(code, oracle)

    def test_decompiled_code_with_struct_access_patterns(self):
        code = """
        typedef long long __int64;
        typedef unsigned long _QWORD;
        __int64 __fastcall func0(unsigned int **a1) {
            unsigned int v2; // [rsp+14h] [rbp-4h]
            if (!*a1)
                return 0LL;
            v2 = **a1;
            if (*a1 == a1[1]) {
                a1[1] = 0LL;
                *a1 = a1[1];
            } else {
                *a1 = (unsigned int *)*((_QWORD *)*a1 + 1);
                *((_QWORD *)*a1 + 2) = 0LL;
            }
            return v2;
        }
        """
        uint = PRIMITIVE_TYPES["unsigned int"]
        ulong = PRIMITIVE_TYPES["unsigned long"]
        oracle = [
            [Pointer(uint), _int, None], # if (!*a1)
            [None], # return 0LL;
            [Pointer(uint), uint, Pointer(uint), Pointer(uint), _int, None], # v2 = **a1; if (*a1 == a1[1])
            [
                Pointer(uint), # a1[1]
                Pointer(uint), # store
                Pointer(uint), # a1[1]
                Pointer(uint), # *a1
                Pointer(uint), # store
            ],
            [# First statement
                Pointer(uint), # (lhs) *a1
                Pointer(uint), # (rhs) *a1
                Pointer(ulong), # (_QWORD *)
                Pointer(ulong), # + 1
                ulong, # * in *((_QWORD *)*a1 + 1)
                Pointer(uint), # (unsigned int *)
                Pointer(uint), # store
             # Second statement
                Pointer(uint), # *a1
                Pointer(ulong), # (_QWORD *)
                Pointer(ulong), # + 2
                ulong, # * in *((_QWORD *)*a1 + 2)
                ulong # store
            ],
            [None] # return v2
        ]
        self.check_inferred_types(code, oracle)
