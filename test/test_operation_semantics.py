"""Test operation semantics individually.
"""

import unittest

import z3

from .utils import TestValues
from faultless.ir import *
from faultless.c import PRIMITIVE_TYPES

_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]
_char = PRIMITIVE_TYPES["char"]
_uchar = PRIMITIVE_TYPES["unsigned char"]
_int = PRIMITIVE_TYPES["int"]
_long = PRIMITIVE_TYPES["long"]
_ulong = PRIMITIVE_TYPES["unsigned long"]

z3solver = z3.Solver()
class TestOperationSemantics(TestValues):
    def assertValuePairsEqual(self, p1: tuple[AddressableValue | None, Value], p2: tuple[AddressableValue | None, Value]):
        if p1[0] is not None and p2[0] is not None:
            self.assertValuesEqual(p1[0], p2[0])
        elif p1[0] is not None or p2[0] is not None:
            return False
        self.assertValuesEqual(p1[1], p2[1])

    def assertTopWriteEqual(self, memory: AddressMapping[AddressT], base_address: AddressT, oracle_offset: Offset, oracle_value: Value):
        write = memory.mapping[base_address]
        assert isinstance(write, Write), f"Expected a write in memory but found {type(write)}"
        self.assertOffsetsEqual(write.offset, oracle_offset)
        self.assertValuesEqual(write.value, oracle_value)

    def test_integer_plus(self):
        addition = Addition()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            addition.execute([v1, v2]),
            AddressableValue(_int, x_var + 4, Symbol(_int, "x",  x_var, False), ())
        )

        y_var = z3.BitVec('y', _long.size * 8)
        v3 = AddressableValue(_long, y_var, Symbol(_long, "y", y_var, False), ())
        self.assertValuesEqual(
            addition.execute([v1, v3]),
            Value(_long, z3.SignExt(y_var.size() - x_var.size(), x_var) + y_var)
        )

        z_var = z3.BitVec('z', 8)
        v4 = AddressableValue(_uchar, z_var, Symbol(_uchar, 'z', z_var, False), ())
        self.assertValuesEqual(
            addition.execute([v3, v4]),
            Value(_long, y_var + z3.ZeroExt(y_var.size() - z_var.size(), z_var))
        )

        w_var = z3.BitVec('w', _ulong.size * 8)
        v5 = AddressableValue(_ulong, w_var, Symbol(_ulong, "w", w_var, False), ())
        self.assertValuesEqual(
            addition.execute([v1, v5]),
            Value(_ulong, z3.SignExt(w_var.size() - x_var.size(), x_var) + w_var)
        )

        arr = Value.make((Array(_int, 3), "arr"))
        self.assertValuesEqual(
            addition.execute([arr, Value.make(IntegerConstant(1, _long))]),
            AddressableValue(Pointer(_int), arr.expr + 4, arr.base_address, ())
        )
    
    def test_integer_minus(self):
        subtraction = Subtraction()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            subtraction.execute([v1, v2]),
            AddressableValue(_int, x_var - 4, Symbol(_int, "x",  x_var, False), ())
        )

        y_var = z3.BitVec('y', _long.size * 8)
        v3 = AddressableValue(_long, y_var, Symbol(_long, "y", y_var, False), ())
        self.assertValuesEqual(
            subtraction.execute([v3, v1]),
            Value(_long, y_var - z3.SignExt(y_var.size() - x_var.size(), x_var))
        )

        z_var = z3.BitVec('z', 8)
        v4 = AddressableValue(_uchar, z_var, Symbol(_uchar, 'z', z_var, False), ())
        self.assertValuesEqual(
            subtraction.execute([v1, v4]),
            Value(_int, x_var - z3.ZeroExt(x_var.size() - z_var.size(), z_var))
        )

        ptr_var = z3.Int('ptr')
        ptr = AddressableValue(Pointer(_int), ptr_var, Symbol(Pointer(_int), "ptr", ptr_var, False), ())
        self.assertValuesEqual(
            subtraction.execute([ptr, v1]),
            AddressableValue(Pointer(_int), ptr_var - 4 * z3.BV2Int(x_var, is_signed=True), Symbol(Pointer(_int), "ptr", ptr_var, False), ())
        )

        arr = Value.make((Array(_int, 3), "arr"))
        self.assertValuesEqual(
            subtraction.execute([arr, Value.make(IntegerConstant(1, _long))]),
            AddressableValue(Pointer(_int), arr.expr - 4, arr.base_address, ())
        )

        arr = Variable(Array(_int, 32), "arr")
        left = Value.make(arr)
        right = Value.make(arr)
        left.expr = left.expr + 12
        right.expr = right.expr + 4
        self.assertValuesEqual(
            subtraction.execute([left, right]),
            Value.make(IntegerConstant(2, SIZE_T))
        )

        other = Value.make(Variable(Array(_int, 32), "xs"))
        with self.assertRaises(SemanticError):
            subtraction.execute([left, other])
    
    def test_integer_multiply(self):
        multiplication = Multiplication()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))

        # int * int -> int and result is an rvalue
        self.assertValuesEqual(
            multiplication.execute([v1, v2]),
            Value(_int, x_var * 4)
        )

        # char operands are integer-promoted to int before multiplication
        c1_var = z3.BitVec('c1', _char.size * 8)
        v3 = AddressableValue(_char, c1_var, Symbol(_char, "c1", c1_var, False), ())
        c2_val = z3.BitVecVal(2, _char.size * 8)
        v4 = Value(_char, c2_val)
        self.assertValuesEqual(
            multiplication.execute([v3, v4]),
            Value(_int, z3.SignExt((_int.size - _char.size) * 8, c1_var) * z3.SignExt((_int.size - _char.size) * 8, c2_val))
        )

    def test_integer_divide(self):
        division = Division()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            division.execute([v1, v2]),
            Value(_int, x_var / 4)
        )

        y_var = z3.BitVec('y', _long.size * 8)
        v3 = AddressableValue(_long, y_var, Symbol(_long, "y", y_var, False), ())
        const200 = z3.BitVecVal(200, _uchar.size * 8)
        v4 = Value(_uchar, const200)
        self.assertValuesEqual(
            division.execute([v3, v4]),
            Value(_long, y_var / z3.ZeroExt(y_var.size() - const200.size(), const200))
        )

    def test_integer_modulus(self):
        modulus = ModulusDivision()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(5, _int.size * 8))
        self.assertValuesEqual(
            modulus.execute([v1, v2]),
            Value(_int, x_var % 5)
        )

        w_var = z3.BitVec('w', _ulong.size * 8)
        v3 = AddressableValue(_ulong, w_var, Symbol(_ulong, "w", w_var, False), ())
        const7 = z3.BitVecVal(7, _int.size * 8)
        v4 = Value(_int, const7)
        self.assertValuesEqual(
            modulus.execute([v3, v4]),
            Value(_ulong, w_var % z3.SignExt(w_var.size() - const7.size(), const7))
        )

        c1_var = z3.BitVec('c1', _char.size * 8)
        c2_val = z3.BitVecVal(255, _uchar.size * 8)
        self.assertValuesEqual(
            modulus.execute([AddressableValue(_char, c1_var, Symbol(_char, "c1", c1_var, False), ()), Value(_uchar, c2_val)]),
            Value(_int, z3.SignExt((_int.size - _char.size) * 8, c1_var) % z3.ZeroExt((_int.size - _uchar.size) * 8, c2_val))
        )

    def test_sizeof(self):
        sizeof = SizeOf()

        self.assertValuesEqual(
            sizeof.execute([_int]),
            Value.make(IntegerConstant(_int.get_size(), SIZE_T))
        )

        self.assertValuesEqual(
            sizeof.execute([Value.make((Array(_int, 3), "arr"))]),
            Value.make(IntegerConstant(Array(_int, 3).get_size(), SIZE_T))
        )

        self.assertValuesEqual(
            sizeof.execute([Array(_int, 7)]), # type: ignore
            Value.make(IntegerConstant(Array(_int, 7).get_size(), SIZE_T))
        )

        self.assertValuesEqual(
            sizeof.execute([Pointer(_char)]), # type: ignore
            Value.make(IntegerConstant(Pointer(_char).get_size(), SIZE_T))
        )

        self.assertValuesEqual(
            sizeof.execute([Value.make(ZERO)]),
            Value.make(IntegerConstant(_int.get_size(), SIZE_T))
        )

    def test_integer_bitshift_left(self):
        left_shift = LeftShift()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(2, _int.size * 8))
        self.assertValuesEqual(
            left_shift.execute([v1, v2]),
            Value(_int, x_var << 2)
        )

    def test_integer_bitshift_right(self):
        right_shift = RightShift()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(2, _int.size * 8))
        self.assertValuesEqual(
            right_shift.execute([v1, v2]),
            Value(_int, x_var >> 2)
        )

        u_var = z3.BitVec('u', _ulong.size * 8)
        v3 = AddressableValue(_ulong, u_var, Symbol(_ulong, "u", u_var, False), ())
        self.assertValuesEqual(
            right_shift.execute([v3, v2]),
            Value(_ulong, z3.LShR(u_var, 2))
        )

    def test_bitwise_and(self):
        bitwise_and = BitwiseAnd()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(6, _int.size * 8))
        self.assertValuesEqual(
            bitwise_and.execute([v1, v2]),
            Value(_int, x_var & 6)
        )

    def test_bitwise_or(self):
        bitwise_or = BitwiseOr()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(5, _int.size * 8))
        self.assertValuesEqual(
            bitwise_or.execute([v1, v2]),
            Value(_int, x_var | 5)
        )

    def test_bitwise_xor(self):
        bitwise_xor = BitwiseXOr()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(9, _int.size * 8))
        self.assertValuesEqual(
            bitwise_xor.execute([v1, v2]),
            Value(_int, x_var ^ 9)
        )

    def test_bitwise_not(self):
        bitwise_not = BitwiseNot()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        self.assertValuesEqual(
            bitwise_not.execute([v1]),
            Value(_int, ~x_var)
        )

    def test_unary_minus(self):
        unary_minus = UnaryMinus()

        x_var = z3.BitVec('x_minus', _int.size * 8)
        self.assertValuesEqual(
            unary_minus.execute([AddressableValue(_int, x_var, Symbol(_int, "x_minus", x_var, False), ())]),
            Value(_int, -x_var)
        )

        c_var = z3.BitVec('c_minus', _char.size * 8)
        self.assertValuesEqual(
            unary_minus.execute([AddressableValue(_char, c_var, Symbol(_char, "c_minus", c_var, False), ())]),
            Value(_int, -z3.SignExt((_int.size - _char.size) * 8, c_var))
        )

        with z3repr_options(integer_repr="int", float_repr="real"):
            i_var = z3.Int('i_minus')
            self.assertValuesEqual(
                unary_minus.execute([Value(_int, i_var)]),
                Value(_int, -i_var)
            )

            f_var = z3.Real('f_minus')
            self.assertValuesEqual(
                unary_minus.execute([Value(_float, f_var)]),
                Value(_float, -f_var)
            )

        with z3repr_options(float_repr="float"):
            f_var = z3.FP('fp_minus', z3.Float32())
            self.assertValuesEqual(
                unary_minus.execute([Value(_float, f_var)]),
                Value(_float, -f_var)
            )

    def test_logical_not(self):
        logical_not = LogicalNot()

        x_var = z3.BitVec('x', _int.size * 8)
        intval = AddressableValue(_int, x_var, Symbol(_int, "x", x_var, False), ())
        self.assertValuesEqual(
            logical_not.execute([intval]),
            ConditionalValue(x_var == 0)
        )

        ptr_var = z3.Int('ptr')
        ptrval = AddressableValue(Pointer(_int), ptr_var, Symbol(Pointer(_int), "ptr", ptr_var, False), ())
        self.assertValuesEqual(
            logical_not.execute([ptrval]),
            ConditionalValue(ptr_var == 0)
        )

        with z3repr_options(integer_repr="int"):
            x_var = z3.Int('x_int')
            intval = AddressableValue(_int, x_var, Symbol(_int, "x_int", x_var, False), ())
            self.assertValuesEqual(
                logical_not.execute([intval]),
                ConditionalValue(x_var == 0)
            )

        with z3repr_options(float_repr="real"):
            f_var = z3.Real('f_real')
            floatval = AddressableValue(_double, f_var, Symbol(_double, "f_real", f_var, False), ())
            self.assertValuesEqual(
                logical_not.execute([floatval]),
                ConditionalValue(f_var == 0)
            )

        with z3repr_options(float_repr="float"):
            f_var = z3.FP('f_fp', z3.Float64())
            floatval = AddressableValue(_double, f_var, Symbol(_double, "f_fp", f_var, False), ())
            self.assertValuesEqual(
                logical_not.execute([floatval]),
                ConditionalValue(f_var == z3.FPVal(0.0, z3.Float64()))
            )

    def test_less_than(self):
        less_than = LessThan()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            less_than.execute([v1, v2]),
            ConditionalValue(x_var < 4)
        )

        c_var = z3.BitVec('c', _char.size * 8)
        v3 = AddressableValue(_char, c_var, Symbol(_char, "c", c_var, False), ())
        u_var = z3.BitVec('u', _ulong.size * 8)
        v4 = AddressableValue(_ulong, u_var, Symbol(_ulong, "u", u_var, False), ())
        self.assertValuesEqual(
            less_than.execute([v3, v4]),
            ConditionalValue(z3.ULT(z3.SignExt((_ulong.size - _char.size) * 8, c_var), u_var))
        )

    def test_less_than_or_equal_to(self):
        less_than_or_equal = LessThanOrEqualTo()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            less_than_or_equal.execute([v1, v2]),
            ConditionalValue(x_var <= 4)
        )

        c_var = z3.BitVec('c', _char.size * 8)
        v3 = AddressableValue(_char, c_var, Symbol(_char, "c", c_var, False), ())
        u_var = z3.BitVec('u', _ulong.size * 8)
        v4 = AddressableValue(_ulong, u_var, Symbol(_ulong, "u", u_var, False), ())
        self.assertValuesEqual(
            less_than_or_equal.execute([v3, v4]),
            ConditionalValue(z3.ULE(z3.SignExt((_ulong.size - _char.size) * 8, c_var), u_var))
        )

    def test_greater_than(self):
        greater_than = GreaterThan()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            greater_than.execute([v1, v2]),
            ConditionalValue(x_var > 4)
        )

        c_var = z3.BitVec('c', _char.size * 8)
        v3 = AddressableValue(_char, c_var, Symbol(_char, "c", c_var, False), ())
        u_var = z3.BitVec('u', _ulong.size * 8)
        v4 = AddressableValue(_ulong, u_var, Symbol(_ulong, "u", u_var, False), ())
        self.assertValuesEqual(
            greater_than.execute([v3, v4]),
            ConditionalValue(z3.UGT(z3.SignExt((_ulong.size - _char.size) * 8, c_var), u_var))
        )

    def test_greater_than_or_equal_to(self):
        greater_than_or_equal = GreaterThanOrEqualTo()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            greater_than_or_equal.execute([v1, v2]),
            ConditionalValue(x_var >= 4)
        )

        c_var = z3.BitVec('c', _char.size * 8)
        v3 = AddressableValue(_char, c_var, Symbol(_char, "c", c_var, False), ())
        u_var = z3.BitVec('u', _ulong.size * 8)
        v4 = AddressableValue(_ulong, u_var, Symbol(_ulong, "u", u_var, False), ())
        self.assertValuesEqual(
            greater_than_or_equal.execute([v3, v4]),
            ConditionalValue(z3.UGE(z3.SignExt((_ulong.size - _char.size) * 8, c_var), u_var))
        )

    def test_equal_to(self):
        equal_to = EqualTo()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            equal_to.execute([v1, v2]),
            ConditionalValue(x_var == 4)
        )

        y_var = z3.BitVec('y', _long.size * 8)
        v3 = AddressableValue(_long, y_var, Symbol(_long, "y", y_var, False), ())
        z_var = z3.BitVec('z', _uchar.size * 8)
        v4 = AddressableValue(_uchar, z_var, Symbol(_uchar, "z", z_var, False), ())
        self.assertValuesEqual(
            equal_to.execute([v3, v4]),
            ConditionalValue(y_var == z3.ZeroExt((_long.size - _uchar.size) * 8, z_var))
        )

        z = Value.make((Pointer(_int), "z"))
        v = Value.make((Pointer(Void()), "v"))
        self.assertValuesEqual(equal_to.execute([z, v]), ConditionalValue(z.expr == v.expr))

        self.assertValuesEqual(
            equal_to.execute([z, Value.make(IntegerConstant(0, _int))]),
            ConditionalValue(z.expr == 0)
        )
        
    def test_not_equal_to(self):
        not_equal = NotEqualTo()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x",  x_var, False), ())
        v2 = Value(_int, z3.BitVecVal(4, _int.size * 8))
        self.assertValuesEqual(
            not_equal.execute([v1, v2]),
            ConditionalValue(x_var != 4)
        )

        y_var = z3.BitVec('y', _long.size * 8)
        v3 = AddressableValue(_long, y_var, Symbol(_long, "y", y_var, False), ())
        z_var = z3.BitVec('z', _uchar.size * 8)
        v4 = AddressableValue(_uchar, z_var, Symbol(_uchar, "z", z_var, False), ())
        self.assertValuesEqual(
            not_equal.execute([v3, v4]),
            ConditionalValue(y_var != z3.ZeroExt((_long.size - _uchar.size) * 8, z_var))
        )

        z = Value.make((Pointer(_int), "z"))
        self.assertValuesEqual(
            not_equal.execute([Value.make(IntegerConstant(0, _int)), z]),
            ConditionalValue(0 != z.expr)
        )

    def test_logical_and_or(self):
        logical_and = LogicalAnd()
        logical_or = LogicalOr()

        x = z3.BitVec('x', _int.size * 8)
        y = z3.BitVec('y', _int.size * 8)
        cx = ConditionalValue(x > 4)
        cy = ConditionalValue(y < 2)

        # Nonzero truthiness captured by passing ConditionalValue operands
        self.assertValuesEqual(
            logical_and.execute([cx, cy]),
            ConditionalValue(z3.And(x > 4, y < 2)) # type: ignore
        )
        self.assertValuesEqual(
            logical_or.execute([cx, cy]),
            ConditionalValue(z3.Or(x > 4, y < 2)) # type: ignore
        )

        # Mixed ConditionalValue and raw Value
        vy = Value(_int, y)
        cond = ConditionalValue(x > 7)
        self.assertValuesEqual(
            logical_and.execute([cond, vy]),
            ConditionalValue(z3.And(cond.condition, vy.expr != 0)) # type: ignore
        )

        with z3repr_options(float_repr="real"):
            xf = z3.Real('xf_real')
            yf = z3.Real('yf_real')
            self.assertValuesEqual(
                logical_or.execute([Value(_double, xf), Value(_double, yf)]),
                ConditionalValue(z3.Or(xf != 0, yf != 0)) # type: ignore
            )

        with z3repr_options(float_repr="float"):
            xf = z3.FP('xf_fp', z3.Float64())
            yf = z3.FP('yf_fp', z3.Float64())
            zero = z3.FPVal(0.0, z3.Float64())
            self.assertValuesEqual(
                logical_or.execute([Value(_double, xf), Value(_double, yf)]),
                ConditionalValue(z3.Or(xf != zero, yf != zero)) # type: ignore
            )

    def test_cast_execute(self):
        cast_op = Cast()
        x_var = z3.BitVec('x', _char.size * 8)
        v_char = AddressableValue(_char, x_var, Symbol(_char, "x", x_var, False), ())

        # char -> int should sign-extend. Also, should preserve addressability
        self.assertValuesEqual(
            cast_op.execute([_int, v_char]),
            AddressableValue(_int, z3.SignExt((_int.size - _char.size) * 8, x_var), Symbol(_char, "x", x_var, False), ())
        )

        # int -> long should sign-extend for signed ints
        i_var = z3.BitVec('i', _int.size * 8)
        v_int = Value(_int, i_var)
        self.assertValuesEqual(
            cast_op.execute([_long, v_int]),
            Value(_long, z3.SignExt((_long.size - _int.size) * 8, i_var))
        )

        # long -> int should truncate the upper bits
        l_var = z3.BitVec('l', _long.size * 8)
        v_long = Value(_long, l_var)
        self.assertValuesEqual(
            cast_op.execute([_int, v_long]),
            Value(_int, z3.Extract(_int.size * 8 - 1, 0, l_var)) # type: ignore
        )

        # int -> long -> int
        self.assertValuesEqual(
            cast_op.execute([_int, cast_op.execute([_long, v_int])]),
            v_int
        )

        with z3repr_options(integer_repr="int", float_repr="real"):
            i_real = z3.Int('i_real')
            self.assertValuesEqual(
                cast_op.execute([_double, Value(_int, i_real)]),
                Value(_double, z3.ToReal(i_real)) # type: ignore
            )

            d_real = z3.Real('d_real')
            self.assertValuesEqual(
                cast_op.execute([_int, Value(_double, d_real)]),
                Value(_int, z3.If(d_real >= 0, z3.ToInt(d_real), -z3.ToInt(-d_real))) # type: ignore
            )

        with z3repr_options(float_repr="float"):
            i_bv = z3.BitVec('i_bv', _int.size * 8)
            self.assertValuesEqual(
                cast_op.execute([_double, Value(_int, i_bv)]),
                Value(_double, z3.fpSignedToFP(z3.RNE(), i_bv, z3.Float64()))
            )

            d_fp = z3.FP('d_fp', z3.Float64())
            self.assertValuesEqual(
                cast_op.execute([_float, Value(_double, d_fp)]),
                Value(_float, z3.fpToFP(z3.RNE(), d_fp, z3.Float32()))
            )

    def test_copy_execute(self):
        copy_op = Copy()
        x_var = z3.BitVec('x', _int.size * 8)
        v1 = AddressableValue(_int, x_var, Symbol(_int, "x", x_var, False), ())
        self.assertValuesEqual(
            copy_op.execute([v1]),
            v1
        )
    
    def test_address_of(self):
        addressof = AddressOf()
        heap = Heap()
        stack = Stack()

        x_sym = z3repr((_int, "x"))
        x_val = AddressableValue(_int, x_sym, Symbol(_int, "x", x_sym, False), ())
        xaddr_sym = z3repr(Variable(_int, "x"))
        xaddr_val = AddressableValue(Pointer(_int), xaddr_sym, Variable(_int, "x"), ())

        # &x;
        self.assertValuePairsEqual(
            addressof.execute([x_val], lval=xaddr_val, stack=stack, heap=heap, condition=True),
            (xaddr_val, xaddr_val)
        )

    def test_dereference(self):
        dereference = Dereference()
        heap = Heap()
        stack = Stack()

        # int *x; *x;
        x_sym = z3repr((Pointer(_int), "x"))
        x_val = AddressableValue(Pointer(_int), x_sym, Symbol(_int, "x", x_sym, False), ())
        xaddr_sym = z3repr(Variable(Pointer(_int), "x"))
        xaddr_val = AddressableValue(Pointer(Pointer(_int)), xaddr_sym, Variable(Pointer(_int), "x"), ())
        
        fresh_val = z3repr((_int, "x[0]"))
        self.assertValuePairsEqual(
            dereference.execute([x_val], lval=xaddr_val, stack=stack, heap=heap, condition=True),
            (x_val, AddressableValue(_int, fresh_val, Symbol(_int, "x[0]", fresh_val, False), ()))
        )

        # *(++x);
        x_p_4_val = AddressableValue(Pointer(_int), x_sym + 4,  Symbol(_int, "x", x_sym, False), ())
        fresh_val = z3repr((_int, "x[4]")) # four because ints are four bytes.
        self.assertValuePairsEqual(
            dereference.execute([x_p_4_val], lval=xaddr_val, stack=stack, heap=heap, condition=True),
            (x_p_4_val, AddressableValue(_int, fresh_val, Symbol(_int, "x[4]", fresh_val, False), ()))
        )

        # int a[4]; *a;
        a = Variable(Array(_int, 4), "a")
        a_val = Value.make(a)

        aaddr_sym = z3repr(Variable(Pointer(a.type), a.name))
        aaddr_val = AddressableValue(Pointer(a.type), aaddr_sym, a, ())

        fresh_val = Value.make((_int, "a[0]"))
        self.assertValuePairsEqual(
            dereference.execute([a_val], lval=aaddr_val, stack=stack, heap=heap, condition=True),
            (a_val, fresh_val)
        )
    
    def test_subscript(self):
        subscript = Subscript()
        heap = Heap()
        stack = Stack()

        # int *x; x[2];
        x = Variable(Pointer(_int), "x")
        x_sym = z3repr((x.type, "x"))
        x_val = AddressableValue(Pointer(_int), x_sym, Symbol(_int, "x", x_sym, False), ())
        xaddr_sym = z3repr(Variable(Pointer(_int), "x"))
        xaddr_val = AddressableValue(Pointer(Pointer(_int)), xaddr_sym, x, ())

        fresh_sym = z3repr((_int, "x[8]"))
        self.assertValuePairsEqual(
            subscript.execute([x_val, Value(_int, z3.BitVecVal(2, _int.size * 8))], lval=xaddr_val, stack=stack, heap=heap, condition=True),
            (AddressableValue(Pointer(_int), x_sym + 8, x_val.base_address, ()), AddressableValue(_int, fresh_sym, Symbol(_int, "x[8]", fresh_sym, False), ()))
        )

        # int a[4]; a[2];
        a = Variable(Array(_int, 4), "a")
        a_sym = z3repr(a)
        a_val = AddressableValue(a.type, a_sym, a, ())
        aaddr_sym = z3repr(Variable(Pointer(a.type), a.name))
        aaddr_val = AddressableValue(Pointer(a.type), aaddr_sym, a, ())

        fresh_sym = z3repr((_int, "a[8]"))
        self.assertValuePairsEqual(
            subscript.execute([a_val, Value(_int, z3.BitVecVal(2, _int.size * 8))], lval=aaddr_val, heap=heap, stack=stack, condition=True),
            (AddressableValue(Array(_int, 4), aaddr_sym + 8, aaddr_val.base_address, ()), AddressableValue(_int, fresh_sym, Symbol(_int, "a[8]", fresh_sym, False), ()))
        )

    def test_indirect_member_access(self):
        access = MemberAccess(True)
        heap = Heap()
        stack = Stack()
        struct_t = Struct("thing", [UDT.Field(_int, "x"), UDT.Field(_long, "y"), UDT.Field(Array(_int, 5), "a")])
        
        s_sym = z3repr((Pointer(struct_t), "s"))
        s_val = AddressableValue(Pointer(struct_t), s_sym, Symbol(Pointer(struct_t), "s", s_sym, False), ())
        saddr_sym = z3repr(Variable(Pointer(struct_t), "s"))
        saddr_val = AddressableValue(Pointer(Pointer(struct_t)), saddr_sym, Symbol(Pointer(struct_t), "s", saddr_sym, False), ())
        
        # s->y
        fresh_sym = z3repr((_long, "s[8]"))
        self.assertValuePairsEqual(
            access.execute([s_val, FieldValue(Field('y'))], lval=saddr_val, stack=stack, heap=heap, condition=True),
            (AddressableValue(Pointer(_long), s_sym + 8, Symbol(Pointer(struct_t), "s", s_sym, False), (Field("y"),)), AddressableValue(_long, fresh_sym, Symbol(_long, "s[8]", fresh_sym, False), ()))
        )

        # s->a
        self.assertValuePairsEqual(
            access.execute([s_val, FieldValue(Field('a'))], lval=saddr_val, stack=stack, heap=heap, condition=True),
            (AddressableValue(Array(_int, 5), s_sym + 16, s_val.base_address, (Field("a"),)), AddressableValue(Pointer(_int), s_sym + 16, s_val.base_address, (Field('a'),)))
        )


    def test_struct_initializer(self):
        myarr = Struct("myarr", [UDT.Field(Pointer(Void()), "arr"), UDT.Field(_char, "elem_ype"), UDT.Field(_int, "length")])
        initializer = Initializer(myarr, ["length", "arr"])

        malloccall = Value.make((Pointer(Void()), "\\malloc(\\carg0)"))
        thirty_two = Value.make(IntegerConstant(32, _int))

        self.assertValuesEqual(
            initializer.execute([thirty_two, malloccall]),
            CompoundValue(myarr, {0: malloccall, 8: Value.make((IntegerConstant(0, _char))), 12: thirty_two})
        )

    def test_array_initializer(self):
        ints = Array(_int, 8)
        initializer = Initializer(ints)

        zero = Value.make(ZERO)
        one = Value.make(ONE)
        two = Value.make(IntegerConstant(2, _int))
        three = Value.make(IntegerConstant(3, _int))

        self.assertValuesEqual(
            initializer.execute([one, two, three]),
            CompoundValue(ints, {0: one, 4: two, 8: three, 12: zero, 16: zero, 20: zero, 24: zero, 28: zero})
        )

    def test_scalar_initializer(self):
        initializer = Initializer(_int)
        self.assertValuesEqual(initializer.execute([]), Value.make(ZERO)) # When no arguments are passed, zero initialization.
        
        initializer = Initializer(_float)
        two_point_three = Value.make(FloatConstant(2.3, _float))
        self.assertValuesEqual(initializer.execute([two_point_three]), two_point_three)

    def test_heap_pointer_store(self):
        store = Store()
        heap = Heap()
        stack = Stack()

        # int *x; *x = 3;
        x_deref_sym = z3repr((_int, "x[0]"))
        t0_val = AddressableValue(_int, x_deref_sym, Symbol(_int, "x[0]", x_deref_sym, False), ())
        x_sym = z3repr((Pointer(_int), "x"))
        t0_addr = AddressableValue(Pointer(_int), x_sym, Symbol(Pointer(_int), "x", x_sym, False), ())

        bv_three = Value(_int, z3.BitVecVal(3, _int.size * 8))
        self.assertValuePairsEqual(
            store.execute([t0_val, bv_three], lval=t0_addr, stack=stack, heap=heap, condition=True), 
            (t0_addr, bv_three)
        )

        self.assertDictEqual(stack.mapping, {})
        self.assertTopWriteEqual(heap, t0_addr.base_address, Offset(0, True, 4), bv_three)

    def test_stack_pointer_store(self):
        store = Store()
        heap = Heap()
        stack = Stack()

        # int a[4]; a[1] = 3;
        a = Variable(Array(_int, 4), "a")
        a_first_elem_sym = z3repr((_int, "a[4]"))
        t0_val = AddressableValue(_int, a_first_elem_sym, Symbol(_int, "a[4]", a_first_elem_sym, False), ())
        a_sym = z3repr(a)
        t0_addr = AddressableValue(Pointer(_int), a_sym + 4, a, ())

        bv_three = Value(_int, z3.BitVecVal(3, _int.size * 8))

        self.assertValuePairsEqual(
            store.execute([t0_val, bv_three], lval=t0_addr, stack=stack, heap=heap, condition=True),
            (t0_addr, bv_three)
        )

        self.assertDictEqual(heap.mapping, {})
        self.assertTopWriteEqual(stack, a, Offset(4, True, 4), bv_three)
