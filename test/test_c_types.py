"""Test the class that parses code into the C type system.
"""

import unittest
from typing import Sequence

from faultless.c import *

class TestCTypes(unittest.TestCase):
    def parse(self, code: str):
        root = parser.parse(bytes(code, "utf8")).root_node
        assert root.type == "translation_unit"
        assert not root.has_error, str(root)
        return root
    
    def parse_declarations(self, code: str) -> list[tuple[CType, str]]:
        """Parse declarations into CType objects. Returned all declared variables and functions.
        
        Unlike the parse_declarations in Scope, this returns the type and name of the variables.
        """
        scope = Scope()
        root = self.parse(code)
        declarations: list[tuple[CType, str]] = []
        for node in root.children:
            if node.type == "declaration":
                declarations.extend(scope.parse_declaration(node, False))
            else:
                scope.parse_type(node)
                
        return declarations
    
    def compare_declarations(self, code: str, oracle: Sequence[tuple[CType, str]], check_decl_str: bool = False):
        """Parse each declaration present in `code`, and compare it with the manually-written oracle solution.
        
        If check_decl_str is True, also check that the .declaration() product of the type is equal to the code.
        This is only possible when the code is in the canonical form that .declaration() would return and when 
        there's exactly one declaration.
        """
        parsed = self.parse_declarations(code)
        assert len(parsed) == len(oracle), f"{len(parsed)} != {len(oracle)}"
        for parsed_decl, oracle_decl in zip(parsed, oracle):
            self.assertEqual(parsed_decl, oracle_decl)
        if check_decl_str:
            assert len(oracle) == 1, f"Cannot check .declaration() against `code` if more than one declaration is present."
            self.assertEqual(code, oracle[0][0].declaration(oracle[0][1]) + ";")
    
    def test_primitive_types(self):
        code = """
        int x;
        long int y;
        unsigned char alpha;
        float z;
        """

        oracle = [
            (SignedInteger("int", 4), "x"),
            (SignedInteger("long", 8), "y"),
            (UnsignedInteger("unsigned char", 1), "alpha"),
            (Float("float", 4), "z"),
        ]

        self.compare_declarations(code, oracle)

    def test_int_array(self):
        code = "int arr[5];"
        oracle = [(Array(SignedInteger("int", 4), 5), "arr")]
        self.compare_declarations(code, oracle, True)

    def test_long_pointer(self):
        code = "long int *x;"
        oracle = [(Pointer(SignedInteger("long", 8)), "x")]
        self.compare_declarations(code, oracle)
    
    def test_multideclarator_declaration(self):
        code = "int x, *y;"
        oracle = [
            (SignedInteger('int', 4), 'x'), 
            (Pointer(SignedInteger('int', 4)), 'y')
        ]
        self.compare_declarations(code, oracle)

    def test_point_struct(self):
        code = """
        struct point { int x; int y; };
        struct point pt;
        """

        oracle = [
            (Struct("point", [
                UDT.Field(SignedInteger("int", 4), "x"), 
                UDT.Field(SignedInteger("int", 4), "y")
            ]), "pt")
        ]

        self.compare_declarations(code, oracle)

    def test_struct_layout_with_padding(self):
        code = """struct padded { char a; int b; char c; } p;"""

        oracle = [
            (Struct("padded", [
                UDT.Field(SignedInteger("char", 1), "a"),
                UDT.Field(SignedInteger("int", 4), "b"),
                UDT.Field(SignedInteger("char", 1), "c"),
            ]), "p")
        ]

        self.assertEqual(oracle[0][0].fieldname2offset, {"a": 0, "b": 4, "c": 8})
        self.assertEqual(oracle[0][0].get_size(), 12)
        self.assertEqual(oracle[0][0].offsetof(Field("b")), 4)
        self.compare_declarations(code, oracle, check_decl_str=True)

    def test_struct_layout_aligns_flexible_array_member(self):
        code = """struct packet { char tag; int payload[]; } p;"""

        oracle = [
            (Struct("packet", [
                UDT.Field(SignedInteger("char", 1), "tag"),
                UDT.Field(Array(SignedInteger("int", 4), 0), "payload"),
            ]), "p")
        ]

        self.assertEqual(oracle[0][0].fieldname2offset, {"tag": 0, "payload": 4})
        self.assertEqual(oracle[0][0].get_size(), 4)
        self.compare_declarations(code, oracle, check_decl_str=True)

    def test_struct_layout_flexible_array_updates_struct_alignment(self):
        code = """struct packet { char tag; long payload[]; } p;"""

        oracle = [
            (Struct("packet", [
                UDT.Field(SignedInteger("char", 1), "tag"),
                UDT.Field(Array(SignedInteger("long", 8), 0), "payload"),
            ]), "p")
        ]

        self.assertEqual(oracle[0][0].fieldname2offset, {"tag": 0, "payload": 8})
        self.assertEqual(oracle[0][0].get_size(), 8)
        self.compare_declarations(code, oracle, check_decl_str=True)

    def test_simple_union(self):
        code = """union u { long x; unsigned char bytes[8]; } y;"""

        oracle = [
            (Union("u", [
                UDT.Field(SignedInteger("long", 8), "x"),
                UDT.Field(Array(UnsignedInteger("unsigned char", 1), 8), "bytes")
            ]), "y")
        ]
        self.compare_declarations(code, oracle, True)

    def test_out_of_order_nested_struct_expands_on_lookup(self):
        code = """
        struct outer { struct inner value; };
        struct inner { int x; int y; };
        struct outer out;
        """

        inner = Struct("inner", [
            UDT.Field(SignedInteger("int", 4), "x"),
            UDT.Field(SignedInteger("int", 4), "y"),
        ])
        oracle = [
            (Struct("outer", [UDT.Field(inner, "value")]), "out")
        ]
        self.compare_declarations(code, oracle)

    def test_out_of_order_enum_member_expands_on_lookup(self):
        code = """
        struct holder { enum state state; int value; };
        enum state { READY=0, DONE=1 };
        struct holder h;
        """

        state = Enum(name="state", members=[
            Enum.Member(name="READY", value=0),
            Enum.Member(name="DONE", value=1),
        ])
        oracle = [
            (Struct("holder", [
                UDT.Field(state, "state"),
                UDT.Field(SignedInteger("int", 4), "value"),
            ]), "h")
        ]
        self.compare_declarations(code, oracle)

    def test_simple_function(self):
        code = "int foo(char x, unsigned y);"
        oracle = [(FunctionType(SignedInteger("int", 4), [(SignedInteger("char", 1), "x"), (UnsignedInteger("unsigned int", 4), "y")]), "foo")]
        self.compare_declarations(code, oracle) # no decl string due to unsigned vs. unsigned int

    def test_nested_pointers(self):
        code = "int *** x;"
        oracle = [(Pointer(Pointer(Pointer(SignedInteger("int", 4)))), "x")]
        self.compare_declarations(code, oracle) # no decl string to to space after ***.

    def test_array_of_poitners(self):
        # array of pointers to floats
        code="float *xs[5];"
        oracle = [(Array(Pointer(Float("float", 4)), 5), "xs")]
        self.compare_declarations(code, oracle, True)

    def test_array_pointer(self):
        # pointer to an array of floats
        code = "float (*ys)[5];"
        oracle = [(Pointer(Array(Float("float", 4), 5)), "ys")]
        self.compare_declarations(code, oracle, True)

    def test_array_of_array_pointers(self):
        # array of 4 pointers to arrays of three floats
        code = "float (*zs[5])[3];"
        oracle = [(Array(Pointer(Array(Float("float", 4), 3)), 5), "zs")]
        self.compare_declarations(code, oracle, True)

    def test_four_nexted_arrays_of_float_pointers(self):
        code = "float *as[2][3][4][5];"
        oracle = [(Array(Array(Array(Array(Pointer(Float("float", 4)), 5), 4), 3), 2), "as")]
        self.compare_declarations(code, oracle, True)

    def test_function_pointer(self):
        code = "char (*it)(int, int);"
        oracle = [(Pointer(FunctionType(SignedInteger("char", 1), [
            (SignedInteger("int", 4), None),
            (SignedInteger("int", 4), None)
        ])), "it")]
        self.compare_declarations(code, oracle, True)
    
    def test_function_pointer_with_complex_argument(self):
        code = "long long (*get_size)(int (**)[7]);"
        oracle = [(Pointer(FunctionType(SignedInteger("long long", 8), [
            (Pointer(Pointer(Array(SignedInteger("int", 4), 7))), None)
        ])), "get_size")]
        self.compare_declarations(code, oracle, True)

    def test_nested_function_pointers(self):
        code = "char (*predicate)(void **, char [17], int (*)(long, char));"
        oracle = [
            (Pointer(FunctionType(SignedInteger("char", 1), [
                (Pointer(Pointer(Void())), None),
                (Array(SignedInteger("char", 1), 17), None),
                (Pointer(FunctionType(SignedInteger("int", 4), [(SignedInteger("long", 8), None), (SignedInteger("char", 1), None)])), None)
            ])), "predicate")
        ]
        self.compare_declarations(code, oracle, True)

    def test_call_to_typedef_alias(self):
        code = """
        typedef struct {
            int x;
            int y;
        } point;

        typedef point *point_pt;
        void plot(point);
        """
        oracle = [(FunctionType(Void(),
            [(Struct("point", [
                UDT.Field(SignedInteger("int", 4), "x"),
                UDT.Field(SignedInteger("int", 4), "y")
            ]), None)]
        ), "plot")]
        self.compare_declarations(code, oracle)

    def test_deeply_nested_declarators(self):
        code = "char *(*(**bar[][8])())[];"
        oracle = [
            (Array(Array(Pointer(Pointer(FunctionType(Pointer(Array(Pointer(SignedInteger("char", 1)), 0)), []))), 8), 0),
             "bar")
        ]
        self.compare_declarations(code, oracle, True)

    ### Non-declaration tests
    def test_self_referential_struct_keeps_definition_stub_on_recursive_edge(self):
        code = """
        struct node { int value; struct node *next; };
        struct node n;
        """

        parsed = self.parse_declarations(code)
        self.assertEqual(len(parsed), 1)
        node_t, name = parsed[0]
        self.assertEqual(name, "n")
        self.assertIsInstance(node_t, Struct)
        assert isinstance(node_t, Struct)
        next_t = node_t.typeof("next")
        self.assertIsInstance(next_t, Pointer)
        assert isinstance(next_t, Pointer)
        self.assertIsInstance(next_t.target_type, IncompleteStruct)
        assert isinstance(next_t.target_type, IncompleteStruct)
        self.assertIs(next_t.target_type.full_definition, node_t)

        for _ in range(5):
            self.assertIsNotNone(next_t.target_type.full_definition)
            recursive_node_t = next_t.target_type.full_definition
            assert isinstance(recursive_node_t, Struct)
            self.assertIs(recursive_node_t, node_t)
            next_t = recursive_node_t.typeof("next")
            self.assertIsInstance(next_t, Pointer)
            assert isinstance(next_t, Pointer)
            self.assertIsInstance(next_t.target_type, IncompleteStruct)
            assert isinstance(next_t.target_type, IncompleteStruct)

    def test_missing_struct_definition(self):
        code = """
        struct outer { int x; int y; struct inner z; };
        struct outer out;
        """

        with self.assertRaises(TypeNotDefinedError):
            print(self.parse_declarations(code))
        
