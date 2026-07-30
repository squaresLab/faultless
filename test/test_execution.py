"""Checks that the interpreter executes faultless IR correctly.
"""

import z3

from .utils import TestValues

from faultless import execute, EquivalenceOptions
from faultless.ir import Value, AddressableValue, CompoundValue, z3repr, z3repr_options, IntegerConstant, Void, Pointer, Struct, UDT, Field, Integer, SymbolicExpression, SIZE_T
from faultless.c import PRIMITIVE_TYPES

_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]
_char = PRIMITIVE_TYPES["char"]
_uchar = PRIMITIVE_TYPES["unsigned char"]
_int = PRIMITIVE_TYPES["int"]
_long = PRIMITIVE_TYPES["long"]
_ulong = PRIMITIVE_TYPES["unsigned long"]

def intc(value: int, int_t: Integer = _int) -> SymbolicExpression:
    """A utility function to construct the correct symbolic expressions for integer values."""
    return Value.make(IntegerConstant(value, int_t)).expr

def conjoin(constraints: list[z3.BoolRef]) -> z3.BoolRef:
    """A utility function which combines constraints together in an aesthetically pleasing way for easier debugging."""
    if len(constraints) == 0:
        return z3.BoolVal(True)
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return z3.And(*constraints) # type: ignore

class TestReturnValuesEqual(TestValues):
    def assertReturnBehaviorEquivalent(self, code: str, oracle: Value | None, return_constraints: list[z3.BoolRef] = [], options: EquivalenceOptions = EquivalenceOptions()):
        execution = execute(code, equivalence_options=options)
        retval = execution.return_value
        if retval is not None and oracle is not None:
            self.assertValuesEqual(retval, oracle)
            self.assertTrue(*self.z3eq(conjoin(execution.return_constraints), conjoin(return_constraints)))
        else:
            self.assertEqual(retval is None, oracle is None)
    
    def test_plus_one(self):
        code = """
        int plusone(int x) {
           return x + 1;
        }
        """

        oracle = Value.make((_int, "\\param0"))
        oracle.expr = oracle.expr + 1
        
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_quadratic_expression(self):
        code = """
        int quadratic(int x) {
           return x * x + 2 * x + 5
        }
        """

        x = Value.make((_int, "\\param0")).expr
        oracle = Value(_int, x * x + 2 * x + 5) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_sizeof_type(self):
        code = """
        unsigned long foo(void) {
           return sizeof(int);
        }
        """

        self.assertReturnBehaviorEquivalent(code, Value.make(IntegerConstant(_int.get_size(), SIZE_T)))

    def test_if_stack_variable_interaction(self):
        code = """
        long foo(int x) {
            long y;
            if (x < 4) {
               y = 4;
            } else {
               y = 7;
            }
            return y;
        }
        """

        x = Value.make((_int, "\\param0")).expr
        expr = z3.If(x < 4, z3repr(IntegerConstant(4, _long)), z3repr(IntegerConstant(7, _long)))
        oracle = Value(_long, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_if_two_return_paths(self):
        code = """
        long foo(int x) {
            if (x) {
                return 8;
            } else {
                return 0;
            }
        }
        """

        x = Value.make((_int, "\\param0")).expr
        expr = z3.If(x != 0, z3repr(IntegerConstant(8, _long)), z3repr(IntegerConstant(0, _long)))
        oracle = Value(_long, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_nested_if_group_merge(self):
        code = """
        int foo(int *a) {
            int x;
            if (a[0]) {
                if (a[1]) {
                    x = a[1] + a[0]
                } else {
                    x = 4;
                }
            } else {
                if (a[2]) {
                    x = a[2];
                } else {
                    x = 3;
                }
            }
            return x;
        }
        """

        param0 = [Value.make((_int, f"\\param0[{i * 4}]")).expr for i in range(3)]
        expr = z3.If(param0[0] != 0,
            z3.If(param0[1] != 0, param0[1] + param0[0], z3repr(IntegerConstant(4, _int))),
            z3.If(param0[2] != 0, param0[2], z3repr(IntegerConstant(3, _int)))
        )
        oracle = Value(_int, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)
    
    def test_control_flow_inducing_expression(self):
        code = """
        int valid(int *x) {
            return x && *x;
        }
        """

        x = Value.make((Pointer(_int), "\\param0")).expr
        x_deref = Value.make((_int, "\\param0[0]")).expr

        # The execution will have a syntatically different representation of the if condition reflecting the short-circuiting 
        # nature of the logical and operation, but it should be mathematically equivalent to this.
        expr = z3.If(z3.And(x != 0, x_deref != 0), z3repr(IntegerConstant(1, _int)), z3repr(IntegerConstant(0, _int)))
        oracle = Value(_int, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)


    def test_control_flow_inducing_expression_in_if(self):
        code = """
        int foo(int *x) {
            if (x && *x) {
                return *x + 1;
            }
            return -1;
        }
        """

        x = Value.make((Pointer(_int), "\\param0")).expr
        x_deref = Value.make((_int, "\\param0[0]")).expr

        expr = z3.If(z3.And(x != 0, x_deref != 0), x_deref + 1, z3repr(IntegerConstant(-1, _int)))
        oracle = Value(_int, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_basic_for_loop_array_sum(self):
        code = """
        int sum_array(int *arr, int n) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += arr[i];
            }
            return sum;
        }
        """

        oracle = Value.make((_int, f"\\$phi_sum"))
        post_i = Value.make((_int, "\\$phi_i")).expr
        n = Value.make((_int, "\\param1")).expr
        self.assertReturnBehaviorEquivalent(code, oracle, [post_i >= n])

    def test_sum_array(self):
        code = """
        int sum_init_arr(int arr[], int n) {
            arr[0] = 0;
            for (int i = 1; i < n; ++i) {
                arr[i] = arr[i - 1] + 1;
            }
            return arr[n - 1];
        }
        """

        with z3repr_options(integer_repr="int"):
            i = Value.make((_int, "\\phi_i")).expr
            post_i = Value.make((_int, "\\$phi_i")).expr
            n = Value.make((_int, "\\param1")).expr
            lt_memory = Value.make((_int, "\\param0[\\marg3]")).expr
            in_bounds_memory = Value.make((_int, "\\param0[\\marg2]")).expr
            # The commented-out version is ideal but assumes more completeness than we currently have.
            # expr = z3.If(n <= 1, lt_memory, z3.If(n == 1, 0, in_bounds_memory + 1))
            expr = z3.If(z3.Exists([i], z3.And(1 <= i, i < n, i == n - 1)), 1 + z3.If(i == 1, 0, in_bounds_memory), z3.If(n == 1, 0, lt_memory) ) # type: ignore
            oracle = Value(_int, expr) # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle, [post_i >= n])

    def test_function_call(self):
        code = """
        int * bar(int x, void * y, int z);

        int * foo(int x, void * y) {
            bar(x * 34 + 33467, y, x + 2);
            return bar(x * 33 + x + 33467, y, x + 1 + 1);
        }
        """

        with z3repr_options(integer_repr="int"):
            oracle = Value.make((Pointer(_int), "bar(\\carg0, \\param1, \\param0 + 2)"))
            self.assertReturnBehaviorEquivalent(code, oracle)

    def test_float_return_real_repr(self):
        code = """
        double add(double x) {
            return x + 1.5;
        }
        """

        with z3repr_options(float_repr="real"):
            x = Value.make((_double, "\\param0")).expr
            oracle = Value.make((_double, "\\param0"))
            oracle.expr = x + z3.RealVal("1.5") # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle)

    def test_float_return_builtin_fp_repr(self):
        code = """
        double add(double x) {
            return x + 1.5;
        }
        """

        with z3repr_options(float_repr="float"):
            x = Value.make((_double, "\\param0")).expr
            oracle = Value.make((_double, "\\param0"))
            oracle.expr = x + z3.FPVal(1.5, z3.Float64()) # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle)
    
    def test_cast_execution(self):
        code = """long * simplecast(void * x) { return (long *)x; }"""
        oracle = Value.make((Pointer(_long), "\\param0"))
        oracle.base_address.type = Pointer(Void())
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_read_from_function_call_retval(self):
        code = """
        char * stralloc(int len) {
            char * str = (char *)malloc(len);
            if (!str) crash();
            str[len - 1] = 0;
            return str;
        }
        """

        with z3repr_options(integer_repr="int"):
            oracle = Value.make((Pointer(_char), "malloc(\\param0)"))
            oracle.base_address = Value.make((Pointer(Void()), "malloc(\\param0)")).base_address
            self.assertReturnBehaviorEquivalent(code, oracle)

    def test_local_struct_argument_field_access(self):
        code = """
        struct point { int x; int y; };
        int first(struct point pt) {
            return pt.x;
        }
        """

        oracle = Value.make((_int, "\\param0[0]"))
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_returned_struct(self):
        code = """
        struct point { int x; int y; };
        struct point x_component(struct point pt) {
            pt.y = 0;
            return pt;
        }
        """

        point_t = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])
        oracle = CompoundValue.make((point_t, "\\param0"))
        oracle.offset_values[4] = Value.make((IntegerConstant(0, _int)))
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_struct_from_call(self):
        code = """
        struct vector { int x; int y; int z; };
        struct vector origin();
        struct vector invert(struct vector v) {
            struct vector orig = origin();
            v.x = orig.x - v.x;
            v.y = orig.y - v.y;
            v.z = orig.z - v.z;
            return v;
        }
        """

        vector_t = Struct("vector", [UDT.Field(_int, "x"), UDT.Field(_int, "y"), UDT.Field(_int, "z")])
        oracle = CompoundValue.make((vector_t, "\\param0"))
        origin = CompoundValue.make((vector_t, "origin()"))
        oracle.offset_values[0] = Value(_int, origin.offset_values[0].expr - oracle.offset_values[0].expr)
        oracle.offset_values[4] = Value(_int, origin.offset_values[4].expr - oracle.offset_values[4].expr)
        oracle.offset_values[8] = Value(_int, origin.offset_values[8].expr - oracle.offset_values[8].expr)
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_in_expression_struct_field_access(self):
        code = """
        struct point { int x; int y; };
        struct point center();
        int foo(int arg) {
            return (struct point){ .x=4 - arg, .y=3 }.x + center().y;
        }
        """
        
        point_t = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])
        argument = Value.make((_int, "\\param0"))
        center = CompoundValue.make((point_t, "center()"))
        oracle = Value(_int, center.get(Field('y')).expr + 4 - argument.expr)
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_single_element_access_of_manually_initialized_array(self):
        code = """
        int foo(int x) {
            int arr[5];
            arr[2] = 17;
            if (x == 1 || x == 2) {
                return arr[x];
            }
            return 0;
        }
        """

        with z3repr_options(integer_repr="int"):
            x = Value.make((_int, "\\param0"))
            default = Value.make((_int, "arr[4*\\param0]"))
            oracle = Value(_int, z3.If(x.expr == 2, 17, z3.If(x.expr == 1, default.expr, 0))) # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle)

    def test_sum_array_pair(self):
        code = """
        int foo(int z) {
           int a[3] = {z, 99, 98};
           a[2] = a[2] - 93;
           return a[0] + a[2];
        }
        """

        z = Value.make((_int, "\\param0"))
        z.expr = z.expr + 5
        self.assertReturnBehaviorEquivalent(code, z)

    def test_string_literal_access(self):
        code = """
        int stringthings(int x) {
            int y = *"abc";
            if (x >= 0 && x < 3) {
                y -= "AB"[x] + 1;
            }
            return y;
        }
        """
        x = Value.make((_int, "\\param0")).expr
        oracle = Value(_int, z3.If(x == 0, intc(31), z3.If(x == 1, intc(30), z3.If(x == 2, intc(96), intc(97))))) # type: ignore
        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_early_return_nested_ifs(self):
        code = """
        int foo(int x, int y) {
            if (x > 0) {
                if (y > 0) {
                    return x + y;
                }
            }
            return x - y;
        }
        """

        x = Value.make((_int, "\\param0")).expr
        y = Value.make((_int, "\\param1")).expr
        expr = z3.If(x > 0, z3.If(y > 0, x + y, x - y), x - y)
        oracle = Value(_int, expr) # type: ignore

        self.assertReturnBehaviorEquivalent(code, oracle)

    def test_return_in_loop(self):
        code = """
        int foo(int x) {
            int y = x;
            while (y > 0) {
                if (y == 5) {
                    return 7;
                }
                y = y - 1;
            }
            return 4;
        }
        """

        with z3repr_options(integer_repr="int"):
            x = Value.make((_int, "\\param0")).expr
            y = Value.make((_int, "\\$phi_y")).expr
            expr = z3.If(y > 0, 7, 4)
            oracle = Value(_int, expr) # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle, [z3.Implies(y > 0, z3.And(y <= x, y == 5))])

    def test_break_in_loop(self):
        code = """
        int foo(int x) {
            int y = x;
            while (y > 0) {
                y = y - 1;
                if (y == 9) break;
            }
            return y;
        }
        """

        with z3repr_options(integer_repr="int"):
            x = Value.make((_int, "\\param0")).expr
            y = Value.make((_int, "\\$phi_y")).expr
            expr = z3.If(y > 0, y - 1, y)
            oracle = Value(_int, expr) # type: ignore
            self.assertReturnBehaviorEquivalent(code, oracle, [z3.Implies(y > 0, z3.And(y <= x, y - 1 == 9))])
    
    def test_global_access(self):
        code = """
        int foo(int x) {
            return g[1] + x;
        }
        """

        g1 = Value.make((_int, "\\global_g[4]")).expr
        x = Value.make((_int, "\\param0")).expr
        self.assertReturnBehaviorEquivalent(code, Value(_int, g1 + x))

    def test_pass_by_reference_symvars(self):
        code = """
        int bar(int *x) {
            int y;
            foo(x, &y);
            return x[0] + y;
        }
        """
        foox = Value.make((_int, "foo(*\\param0, &y)"))
        fooy = Value.make((_int, "foo(\\param0, *&y)"))
        oracle = Value(_int, foox.expr + fooy.expr)
        self.assertReturnBehaviorEquivalent(code, oracle, options=EquivalenceOptions(use_pass_by_reference_symvars=True))

        code = """int bar(int *x) { foo(x); return x[0]; }"""
        self.assertReturnBehaviorEquivalent(code, Value.make((_int, "\\param0[0]")), options=EquivalenceOptions(use_pass_by_reference_symvars=False))

    def test_assignable_literal_zero_value(self):
        code = """void * getnull(int x, void * p) { if (x) return p; return 0LL; }"""

        x = Value.make((_int, "\\param0")).expr
        p = Value.make((Pointer(Void()), "\\param1")).expr
        self.assertReturnBehaviorEquivalent(code, Value(Pointer(Void()), z3.If(x != 0, p, 0))) # type: ignore

    def test_do_while_loop(self):
        code = """int dowhile(int x) { do { print(x); x++; } while(x < 100); return x; }"""

        with z3repr_options(integer_repr="int"):
            x = Value.make((_int, "\\$phi_x"))
            phi_x = x.expr
            x.expr = x.expr + 1
            self.assertReturnBehaviorEquivalent(code, x, [phi_x >= 99]) # 99 because of the + 1 on the second iteration.

    def test_control_flow_inducing_expressions_in_loop_header(self):
        code = """
        char * and_or(char * s, int n) {
            char * end = s + n;
            while ((*s && s[0] < 63) || s < end) {
                log(s);
                s++;
            }
            return s;
        }
        """

        with z3repr_options(integer_repr="int"):
            s = Value.make((Pointer(_char), "\\param0"))
            phi_s = Value.make((Pointer(_char), "\\$phi_s"))
            phi_s0 = Value.make((_char, "\\$phi_s[0]"))
            n = Value.make((Pointer(_char), "\\param1"))
            end = s.expr + n.expr
            
            self.assertReturnBehaviorEquivalent(code, phi_s, [z3.And(z3.Or(phi_s0.expr == 0, phi_s0.expr >= 63), phi_s.expr >= end)]) # type: ignore -- z3 typing

    def test_intersecting_memory_traversals(self):
        code = """
        char foo(char * p, int cond, int i) {
            if (cond) {
                p[2] = 77;
            } else {
                p[2] = 41;
            }
            p[1] = 100;
            if (cond) {
                p[0] = 39;
            } else {
                p[0] = 24;
            }
            return p[i];
        }
        """

        with z3repr_options(integer_repr="int"):
            fresh = Value.make((_char, "\\param0[\\param2]")).expr
            cond = Value.make((_int, "\\param1")).expr
            i = Value.make((_int, "\\param2")).expr
            expr = z3.If(cond != 0, 
                z3.If(i == 0, 39, z3.If(i == 1, 100, z3.If(i == 2, 77, fresh))),
                z3.If(i == 0, 24, z3.If(i == 1, 100, z3.If(i == 2, 41, fresh)))
            )
            oracle = Value(_char, expr) # type: ignore -- z3 typing
            self.assertReturnBehaviorEquivalent(code, oracle)

    # def test_fibonacci_array_for_loop(self):
    #     code = """
    #     int fibonaccize_array(int *arr, int n) {
    #         arr[0] = 0;
    #         arr[1] = 1;
    #         for (int i = 2; i < n; ++i) {
    #             arr[i] = arr[i - 1] + arr[i - 2];
    #         }
    #         return arr[n - 1];
    #     }
    #     """

    #     with z3repr_options(integer_repr="int"):
    #         # i = Value.make((_int, "\\phi_i")).expr
    #         arr_i_m1 = Value.make((_int, "\\param0[-4 + 4*\\phi_i]")).expr
    #         arr_i_m2 = Value.make((_int, "\\param0[-8 + 4*\\phi_i]")).expr
    #         n = Value.make((_int, "\\param1")).expr

    #         expr = z3.If(n <= 2, 1, z3.If(n == 3, 1 + arr_i_m1, arr_i_m1 + arr_i_m2))
    #         oracle = Value(_int, expr) # type: ignore
    #         self.assertReturnBehaviorEquivalent(code, oracle)

        

        
