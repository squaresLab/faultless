"""Interact with tree_sitter to convert C code into IR form.
"""

from abc import ABC
from typing import overload
import itertools

import tree_sitter_c
from tree_sitter import Language, Parser, Node

from .ir import *

DISABLE_UNION_AND_ENUM_SUPPORT = False

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)

class ParsingError(FaultlessError):
    """tree-sitter could not parse the code without errors."""

class TypeNotFoundError(FaultlessError):
    """There is no type with this name."""

class TypeNotDefinedError(FaultlessError):
    """An incomplete type (stub) exists, but its full definition does not."""



class BlockPointer:
    def __init__(self, block: BasicBlock):
        self.block = block

#
# These classes define how control flows after exiting a nested compound statement block (e.g. the body of a loop).
# By default, flow continues to the next block (indicated by the Next class). However, it may instead jump to an
# arbitrary label with goto (indicated by Label). In a for loop, the the update statement occurs in a block after the
# body. Normal flow (Next) goes to the update statement (e.g. i++), but a break skips it. Similarly, a continue
# statement directs control to the conditional loop-test block, not the update block.
#
class BlockSuccessorAssignment(ABC):
    pass

class Next(BlockSuccessorAssignment):
    pass

class BreakAssignment(BlockSuccessorAssignment):
    pass

class ContinueAssignment(BlockSuccessorAssignment):
    pass

class Label(BlockSuccessorAssignment):
    def __init__(self, label: str):
        self.label = label


################
#     Types    #
################

# TODO: Make this customizable because the size of values can change depending on the platform.
# would also need to change the size value for Pointer.size and Enum.size
PRIMITIVE_TYPES = {
    "void": Void(),
    "char": SignedInteger(name="char", size=1),
    "unsigned char": UnsignedInteger(name="unsigned char", size=1),
    "short": SignedInteger(name="short", size=2),
    "unsigned short": UnsignedInteger(name="unsigned short", size=2),
    "int": SignedInteger(name="int", size=4),
    "unsigned int": UnsignedInteger(name="unsigned int", size=4),
    "long": SignedInteger(name="long", size=8),
    "unsigned long": UnsignedInteger(name="unsigned long", size=8),
    "long long": SignedInteger(name="long long", size=8),
    "unsigned long long": UnsignedInteger(name="unsigned long long", size=8),
    "float": Float(name="float", size=4),
    "double": Float(name="double", size=8),
    "long double": Float(name="long double", size=16),
    "_Bool": UnsignedInteger(name="_Bool", size=1)
}
assert PRIMITIVE_TYPES["int"] == INTEGER
assert PRIMITIVE_TYPES["unsigned long"] == SIZE_T

def generate_primitive_aliases() -> dict[str, str]:
    """Standardize integer type names.
    
    There's more than one way to declare the same integer type in C. For example, 
    "unsigned long" is the same as "long unsigned". This function generates a dictionary that
    converts nonstandard names to the corresponding standard names.
    """
    def valid_with_extra_int(int_t: str) -> bool:
        """Some combinations of int type indentifiers are invalid, like
        int int, int char, signed char int, etc. This function identifiers them.
        """
        return all(invalid not in int_t for invalid in {"int", "char"})
    basic_ints = ["char", "short", "int", "long", "long long"]
    unsigned_ints = ["unsigned " + b for b in basic_ints]
    ints = basic_ints + unsigned_ints
    aliases: dict[str, str] = {}
    for int_t in basic_ints:
        aliases["signed " + int_t] = int_t
        aliases[int_t + " signed"] = int_t
    for int_t in basic_ints:
        aliases[int_t + " unsigned"] = "unsigned " + int_t

    for typename, simplified in itertools.chain(
        ((int_t, int_t) for int_t in ints if valid_with_extra_int(int_t)),
        [alias for alias in aliases.items() if valid_with_extra_int(alias[0])]
    ):
        aliases[typename + " int"] = simplified
    # Special cases that don't fit the patterns above
    aliases["unsigned"] = "unsigned int"
    aliases["signed"] = "int"
    aliases["double long"] = "long double"
    return aliases

PRIMITIVE_ALIASES = generate_primitive_aliases()

# A va_list's implementation is platform dependent. For x86-64, on which the dataset was built,
# the C definition for the implementation is:
# typedef struct {
#    unsigned int gp_offset;
#    unsigned int fp_offset;
#    void *overflow_arg_area;
#    void *reg_save_area;
# } va_list[1];
# The way the typedefs in the headers work out, __builtin_va_list is the "original" name of
# the type and va_list is an alias. We don't include the actual composition of the struct
# because this is an implementation detail that should NOT be relied upon in code.
BUILTINS = {
    "__builtin_va_list": PrimitiveType(name="__builtin_va_list", size=24) # the size is 24 on x86-64, anyway.
}

class Scope:
    ANONYMOUS_COMPOSITE_VALID_PARENTS = {"type_definition", "field_declaration", "declaration", "type_descriptor"}

    def __init__(self, enclosing_scope: 'Scope | None' = None, short_circuit_logical_ops: bool | None = None):
        # Get information from an enclosing scope.
        self.enclosing_scope = enclosing_scope
        if short_circuit_logical_ops is None and enclosing_scope is not None:
            short_circuit_logical_ops = enclosing_scope.short_circuit_logical_ops
        self.short_circuit_logical_ops = bool(short_circuit_logical_ops) if short_circuit_logical_ops is not None else False

        #### Variable and Function Information ####
        # Declared variables
        self.variables: dict[str, Variable] = {}
        # Temporary variables. These are only used inside compound experssions and are passed through
        # the return values of expressions, so we don't need to store the corresponding object here.
        # In particular, they won't share a namespace with declared variables.
        self.temporary_idx = 0
        # Declared functions
        self.functions: dict[str, FunctionType] = {}

        #### Type information ####
        # Maps the name from one type to another, as in a typedef.
        self.aliases: dict[str, str] = PRIMITIVE_ALIASES.copy()
        # Maps the textual name of a type (e.g. "char *") to its object-model representation 
        self.types: dict[str, CType] = PRIMITIVE_TYPES.copy()
        self.types.update(BUILTINS)
        for t in DECOMPILER_PLACEHOLDER_TYPES:
            self.types[t.name] = t
        self.expanded: set[str] = set() # tracks which types have been fully expanded so that we don't have to repeat the work.
        # Maps the textual name of an incomplete UDT to a stub object which can be later mapped back to the full object.
        self.stubs: dict[str, IncompleteUDT] = {}
        # Maps identifiers defined in enums to their values. For enums whose values are expressions, we don't evaluate the expressions and instead assign them to "None"
        self.enum_values: dict[str, int | None] = {}
        # Builtin symbols for which we may or may not have a definition.
        self.builtins: set[str] = set(BUILTINS.keys())

    def variable_exists(self, variable_name: str):
        assert isinstance(variable_name, str)

        if variable_name in self.variables:
            return True
        
        # The variable is not found at the innermost scope. However, it may exist in an outer scope.
        scope = self
        while scope.enclosing_scope is not None:
            scope = scope.enclosing_scope
            if variable_name in scope.variables:
                return True
        
        return False
    
    def check_variable(self, variable_name: str) -> Variable:
        """Check if this variable was defined in this scope or an outer scope. If so, return it; if not,
        add it to the scope.

        :param variable_name: the name of the variable to check
        :returns: the variable object corresponding to this variable.
        """
        assert isinstance(variable_name, str)

        if variable_name in self.variables:
            return self.variables[variable_name]
        
        scope = self
        # The variable is not found at the innermost scope. However, it may exist in an outer scope.
        while scope.enclosing_scope is not None:
            scope = scope.enclosing_scope
            if variable_name in scope.variables:
                return scope.variables[variable_name]
        # This variable name is not defined at any scope.

        # If this variable was not declared, we assume that it was a global variable.
        # We use this approach because this tool is designed to process single functions
        # individually, apart from the rest of the codebase. Thus, we have no way of knowing
        # which identifiers are actually global variables and which are just undeclared identifiers.
        else:
            return scope.add_variable(UnknownType(), variable_name)
    
    def add_parameter(self, typ: CType, name: str) -> Parameter:
        """Add a new parameter variable to this scope.

        :param variable_name: the name of the new parameter variable.
        """
        assert isinstance(name, str)
        p = Parameter(typ, name)
        self.variables[name] = p
        return p

    def create_temporary(self, is_stack_allocated: bool = False):
        """Create a temporary variable with a unique name that does not exist in this scope or 
        any enclosing scope.
        """
        variable = Variable(UnknownType(), f"t{self.temporary_idx}", is_temporary=True, is_stack_allocated=is_stack_allocated)
        self.temporary_idx += 1
        return variable
    
    def add_function(self, typ: FunctionType, name: str):
        if name in self.functions:
            raise SemanticError(f"Function {name} has already been declared at this scope!")
        else:
            self.functions[name] = typ
    
    def add_variable(self, typ: CType, name: str) -> Variable:
        assert not isinstance(typ, FunctionType), f"Function symbols should be added with the add_function method."
        if name in self.variables:
            raise SemanticError(f"Variable {name} has already been declared at this scope!")
        else:
            variable = GlobalVariable(typ, name) if self.enclosing_scope is None else Variable(typ, name)
            self.variables[name] = variable
            return variable

    def add_alias(self, base_name: str, new_name: str):
        assert base_name in self.aliases or base_name in self.types or base_name in self.stubs or base_name in self.builtins, f"Aliasing an unknown symbol {base_name} to {new_name}"
        self.aliases[new_name] = base_name

    def add_stub(self, typ_name: str, stub: IncompleteUDT):
        typ = self.types.get(typ_name)
        if isinstance(stub, IncompleteStruct) and isinstance(typ, Struct):
            stub = IncompleteStruct(stub.name, typ)
        elif isinstance(stub, IncompleteUnion) and isinstance(typ, Union):
            stub = IncompleteUnion(stub.name, typ)
        elif isinstance(stub, IncompleteEnum) and isinstance(typ, Enum):
            stub = IncompleteEnum(stub.name, typ)
        if typ_name in self.stubs:
            assert self.stubs[typ_name] == stub
        self.stubs[typ_name] = stub

    def add_type(self, typ_name: str, typ: CType):
        assert not isinstance(typ, IncompleteType), f"add_type should only be called on fully realized complete types but {typ} is an IncompleteType."
        if isinstance(typ, (Struct, Union, Enum)):
            assert typ.name is not None, f"Cannot add anonymous type {typ} as a globally-available type."
        if typ_name in self.types:
            assert typ == self.types[typ_name], f"Conflicting declarations of type {typ_name}:\n{typ}\n  and\n{self.types[typ_name]}"
        else:
            self.types[typ_name] = typ
        if isinstance(typ, Struct):
            assert typ.name is not None
            self.stubs[typ_name] = IncompleteStruct(typ.name, typ)
        elif isinstance(typ, Union):
            assert typ.name is not None
            self.stubs[typ_name] = IncompleteUnion(typ.name, typ)
        elif isinstance(typ, Enum):
            assert typ.name is not None
            self.stubs[typ_name] = IncompleteEnum(typ.name, typ)

    def add_enum_value(self, name: str, value: Optional[int]):
        """Add record enum values so it is known what they are if they are used in the function.
        Note that :param value: can be None if the enum is initialized with an expression.
        """
        assert name not in self.enum_values
        self.enum_values[name] = value

    def is_builtin(self, symbol: str):
        """Returns true if this symbol is a builtin, or an alias to one.
        """
        while symbol in self.aliases:
            symbol = self.aliases[symbol]
        return symbol in self.builtins

    def get_type(self, typ_name: str, expanded: bool = False) -> CType | None:
        """Get the CType object corresponding to this symbol. If expanded is true, then
        the function is guaranteed to recursively have definitions for all composite types,
        up except recursive definitions, and recursive definitions link back to the full
        type definition.
        """
        while typ_name in self.aliases:
            typ_name = self.aliases[typ_name]
        
        if typ_name in self.types:
            typ = self.types[typ_name]
            if expanded and typ_name not in self.expanded:
                self.types[typ_name] = typ = self.expand_type(typ)
                self.expanded.add(typ_name)
            return typ
        
        if self.enclosing_scope is not None:            
            typ = self.enclosing_scope.get_type(typ_name, expanded=expanded)
            if typ is not None:
                return typ
        
        if typ_name in self.stubs:
            stub = self.stubs[typ_name]
            if expanded:
                # if typ_name is in .expanded, that means .expanded and .types are out of sync.
                assert typ_name not in self.expanded
                self.types[typ_name] = typ = self.expand_type(stub)
                self.expanded.add(typ_name)
                return typ
        return None

    def _get_type_definition(self, typ_name: str) -> CType | None:
        if typ_name in self.types:
            return self.types[typ_name]
        if self.enclosing_scope is not None:
            return self.enclosing_scope._get_type_definition(typ_name)
        return None
    
    def expand_type(self, typ: CType, path: tuple[str, ...] = (), anchors: dict[str, Struct | Union] | None = None) -> CType:
        """Return a copy of typ with known named UDT stubs expanded.

        Recursive references to a struct/union already on the expansion path are
        left as incomplete stubs. For structs, the stub records the definition it
        points back to in struct_definition.
        """
        if anchors is None:
            anchors = {}

        if isinstance(typ, IncompleteStruct):
            typ_name = "struct " + typ.name
            definition = typ.full_definition or self._get_type_definition(typ_name)
            if isinstance(definition, Struct):
                if typ_name in path:
                    complete_type = anchors.get(typ_name, definition)
                    assert isinstance(complete_type, Struct)
                    return IncompleteStruct(typ.name, complete_type)
                return self.expand_type(definition, path, anchors)
            raise TypeNotDefinedError(str(typ))
        
        if isinstance(typ, IncompleteUnion):
            typ_name = "union " + typ.name
            definition = typ.full_definition or self._get_type_definition(typ_name)
            if isinstance(definition, Union):
                if typ_name in path:
                    complete_type = anchors.get(typ_name, definition)
                    assert isinstance(complete_type, Union)
                    return IncompleteUnion(typ.name, complete_type)
                return self.expand_type(definition, path, anchors)
            raise TypeNotDefinedError(str(typ))

        if isinstance(typ, IncompleteEnum):
            typ_name = "enum " + typ.name
            definition = typ.full_definition or self._get_type_definition(typ_name)
            if isinstance(definition, Enum):
                return definition
            raise TypeNotDefinedError(str(typ))
        
        if isinstance(typ, Pointer):
            return Pointer(self.expand_type(typ.target_type, path, anchors))
        
        if isinstance(typ, Array):
            return Array(self.expand_type(typ.element_type, path, anchors), typ.nelements)
        
        if isinstance(typ, FunctionType):
            parameters: list[tuple[CType | FunctionType.VariadicParameter, str | None]] = []
            for param_t, param_name in typ.parameters:
                if isinstance(param_t, FunctionType.VariadicParameter):
                    parameters.append((param_t, param_name))
                else:
                    parameters.append((self.expand_type(param_t, path, anchors), param_name))
            return FunctionType(self.expand_type(typ.return_type, path, anchors), parameters)
        
        if isinstance(typ, Struct):
            typ_name = "struct " + typ.name if typ.name is not None else None
            if typ_name is not None and typ_name in path:
                return IncompleteStruct(typ.name, anchors.get(typ_name, typ)) # type: ignore
            member_path = path + ((typ_name,) if typ_name is not None else ())
            if typ_name is not None:
                expanded = object.__new__(Struct)
                anchors = anchors | {typ_name: expanded}
            else:
                expanded = None
            members: list[UDT.Field] = []
            for member in typ.members:
                members.append(UDT.Field(self.expand_type(member.type, member_path, anchors), member.name))
            if expanded is not None:
                Struct.__init__(expanded, typ.name, members)
                return expanded
            return Struct(typ.name, members)
        
        if isinstance(typ, Union):
            typ_name = "union " + typ.name if typ.name is not None else None
            if typ_name is not None and typ_name in path:
                return IncompleteUnion(typ.name, anchors.get(typ_name, typ)) # type: ignore
            member_path = path + ((typ_name,) if typ_name is not None else ())
            if typ_name is not None:
                expanded = object.__new__(Union)
                anchors = anchors | {typ_name: expanded}
            else:
                expanded = None
            members = [UDT.Field(self.expand_type(member.type, member_path, anchors), member.name) for member in typ.members]
            if expanded is not None:
                Union.__init__(expanded, typ.name, members)
                return expanded
            return Union(typ.name, members)
        
        return typ
    
    def get_function(self, name: str) -> FunctionType | None:
        scope = self
        while scope is not None:
            if name in scope.functions:
                expanded = self.expand_type(scope.functions[name])
                assert isinstance(expanded, FunctionType)
                return expanded
            scope = scope.enclosing_scope
        return None

    def parse_type(self, node: Node) -> Optional[CType]:
        if node.type == "type_definition":
            return self.parse_typedef(node) # returns None
        if node.type == "struct_specifier":
            return self.parse_struct(node)
        if node.type == "union_specifier":
            return self.parse_union(node)
        if node.type == "enum_specifier":
            return self.parse_enum(node)
        
        return None
    
    def record_item(self, node: Node):
        if node.type == "declaration":
            self.parse_declaration(node, False)
        if node.type == "function_definition":
            self.parse_function_signature(node)
        self.parse_type(node)

    def get_or_parse_from_type_descriptor(self, node: Node, expanded: bool = False) -> CType:
        """Parse types that occur in code independently of a declaration: there are
        no named objects (e.g. variables, functions) declared, and thus no variable
        names are returned. If there should be a variable name, use parse_declaration().

        This method will first check to see if the type has already been defined.
        If so, it will fetch the corresponding definition. If not, it will parse it
        from scratch. Finally, it'll apply any abstract declarators that occur.

        :param node: a "type_descriptor" node.
        """
        assert node.type == "type_descriptor", f"Expected abstract type descriptor but found {node.type}"
        base_type_node = get_child(node, "type")
        cast_type = self.get_type(get_text(base_type_node), expanded=expanded)
        if cast_type is None:
            cast_type = self.parse_type(base_type_node)
        if cast_type is None:
            raise TypeNotFoundError(f"Cannot find or parse type for type descriptor {get_text(node)} ({node.start_point})")
        if declarator_node := node.child_by_field_name("declarator"):
            # A type descriptor is present inside of certain operations, like a cast, sizeof, or compound literal expression.
            # In all of these cases we need the full expanded type.
            cast_type, _ = self.parse_abstract_declarators(declarator_node, cast_type, expanded=True)
        return cast_type
    
    def parse_typedef(self, node: Node):
        """Parse a member of a typedef into either an alias or a new type in the TypeInfo object model, depending on the composition of that typedef
        
        :param node: a "type_definition" node
        """
        typ_node = get_child(node, "type")
        declarator = get_child(node, "declarator")

        if typ_node.type in {"sized_type_specifier", "type_identifier", "primitive_type"}:
            typ_text = get_text(typ_node); 
            original_type = self.get_type(typ_text)
            if original_type is None and typ_node.type == "type_identifier":
                # HACK: check for builtins by checking that the builtin starts with a specific string.
                name = get_text(typ_node)
                assert "__builtin" == name[:9] or self.is_builtin(name), f"Unknown identifier {name} is not a builtin."
                self.builtins.add(name)
                assert declarator.type == "type_identifier", f"Typedefs to non-identifier declarators are unsupported for builtins. declarator={declarator.type}, typedef={get_text(node)}"
                new_name = get_text(declarator)
                self.add_alias(name, new_name)
                return
        else:
            assert typ_node.type in {"struct_specifier", "enum_specifier", "union_specifier"}
            # NOTE: to change this to have pointers to user-defined types not expanded to their definitions, remove self.get_type here.
            # You'll have to follow the alias chain, however.
            original_type = self.get_type(get_text(typ_node))
            if original_type is None or isinstance(original_type, IncompleteUDT):
                original_type = self.parse_type(typ_node)
        assert original_type is not None, f"Unknown type node type, undeclared identifier, or anonymous type: {typ_node.type}: {get_text(typ_node)}"

        if isinstance(original_type, PrimitiveType) and declarator.type in {"type_identifier", "primitive_type"}:
            assert original_type.name is not None, f"Type name for {original_type} in typedef {get_text(node)} is None!"
            self.add_alias(original_type.name, get_text(declarator))
        else:
            # We only require types to be fully expanded at or inside a function definition.
            typ, name = self.parse_declarators(declarator, original_type, expanded=False)

            # In a typedef like
            #     typedef struct { int a; int b; } mystruct;
            # the struct object we get back from parse_type() will be anonymous and have a placeholder name.
            # The actual name is the name in the declarator.
            if isinstance(typ, UDT) and original_type.name is None: # type: ignore
                assert isinstance(original_type, UDT), f"Can only set the name of a UDT "
                typ.name = name
            
            if isinstance(typ, IncompleteUDT):
                self.add_stub(name, typ)
            else:
                self.add_type(name, typ)

    def _parse_member(self, node: Node) -> UDT.Field | Struct | Union:
        """Convert a member of a stucture or union in to Field object in the TypeInfo object model.
        If the field is an anonymous struct or union, return that struct or union instead.

        :param node: a "field_declaration" node
        """
        assert node.type == "field_declaration"
        type_node = get_child(node, "type")
        # NOTE: To change this so that pointers to composite types don't contain the definition of that type, remove the call to get_type here and only call parse_type.
        # This will make it so that structs/unions/enums are parsed into a stub form.
        # You'll have to follow the alias chain, however.
        base_type = self.get_type(get_text(type_node))
        if base_type is None:
            base_type = self.parse_type(type_node)
        # TODO: If this type is defined as part of a local variable declaration, the type may actually
        # be present in a scope enclosing this one. In this case, we'll throw an exception, but we could
        # actually get the type. (However, this is rare.)
        if base_type is None:
            raise TypeNotFoundError(f"A type for {get_text(type_node)} of struct/union member {get_text(node)} is not defined.")

        declarator = node.child_by_field_name("declarator")
        if declarator is not None: # normal 'int x' or 'struct pt p' declaration.
            # Don't expand type definitions here. This parameter forces expansion and fails with an exception
            # if it is not possible. This method is frequently called when parsing UDT pre-function UDT definitions.
            # This function will still return an expanded output when expanded=False if each of the types is fully
            # defined but it would be a problem for out-of-order structs inside a function (illegal in C so we could
            # genuinely reject those anyway) or recursively defined structs that are defined inside of a declaration,
            # but that is rare because it is mostly pretty useless.
            typ, field_name = self.parse_declarators(get_child(node, "declarator"), base_type, expanded=False)
            return UDT.Field(typ, field_name)
        elif isinstance(base_type, (Union, Struct)): # An anonymous struct or union nested inside another struct or union
            assert base_type.name is None, f"Non-anonymous union member has no declarator in field declaration '{get_text(node)}'"
            return base_type
        else:
            raise ValueError(f"Unrecognized field declaration format: no declarators and base type of type {type(base_type)} ({base_type})")

    def parse_struct(self, node: Node) -> Struct | IncompleteStruct:
        """Convert a node representing a structure into a TypeInfo object.

        :param node: a "struct_specifier" node.
        """
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type e.g. struct thing (with no { ... } defining the fields.)
            assert typ_identifier and typ_identifier.type == "type_identifier"
            struct_name = get_text(typ_identifier)
            typ = IncompleteStruct(struct_name)
            self.add_stub("struct " + struct_name, typ)
        else:
            fields = []
            assert body.type == "field_declaration_list", f"Struct {get_text(node)} has body of type {body.type}"
            for field in remove_curly_braces(body.children):
                fields.append(self._parse_member(field))
            if typ_identifier is None:
                assert node.parent and node.parent.type in Scope.ANONYMOUS_COMPOSITE_VALID_PARENTS, f"Invalid parent for anonymous struct: {node.parent}"
                typ = Struct(name=None, members=fields)
            else:
                struct_name = get_text(typ_identifier)
                typ = Struct(name=struct_name, members=fields, defer_layout=True)
                self.add_type("struct " + struct_name, typ)
        
        return typ
    
    def parse_union(self, node: Node) -> Union | IncompleteUnion:
        if DISABLE_UNION_AND_ENUM_SUPPORT:
            raise UnsupportedFeatureError(f"Unions are not supported.")
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type e.g. union thing (with no { ... } defining the fields.)
            assert typ_identifier and typ_identifier.type == "type_identifier"
            union_name = get_text(typ_identifier)
            typ = IncompleteUnion(union_name)
            self.add_stub("union " + union_name, typ)
        else:
            fields = []
            assert body.type == "field_declaration_list", f"Union {get_text(node)} has body of type {body.type}"
            for field in remove_curly_braces(body.children):
                fields.append(self._parse_member(field))
            if typ_identifier is None:
                assert node.parent and node.parent.type in Scope.ANONYMOUS_COMPOSITE_VALID_PARENTS, f"Invalid parent for anonymous union: {node.parent}"
                typ = Union(name=None, members=fields)
            else:
                union_name = get_text(typ_identifier)
                typ = Union(name=union_name, members=fields)
                self.add_type("union " + union_name, typ)
        
        return typ

    def parse_enum(self, node: Node) -> Enum | IncompleteEnum | None:
        if DISABLE_UNION_AND_ENUM_SUPPORT:
            raise UnsupportedFeatureError(f"Enums are not supported.")
        typ_identifier = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None: # An incomplete type
            assert typ_identifier and typ_identifier.type == "type_identifier"
            enum_name = get_text(typ_identifier)
            typ = IncompleteEnum(enum_name)
            self.add_stub("enum " + enum_name, typ)
        else:
            members = []
            value = 0 # values implicitly start at zero and are incremented, unless otherwise specified.
            assert body.type == "enumerator_list"
            for enumerator in remove_curly_braces(body.children):
                if enumerator.type == ",":
                    continue
                assert enumerator.type == "enumerator", f"Found non-enumerator type in enum!"
                value_node = enumerator.child_by_field_name("value")
                if value_node:
                    if value_node.type == "number_literal":
                        value = parse_int(get_text(value_node))
                    else: # value is some expression. We could try to execute it to get the value, but that could be difficult and possibly dangerous
                        value = None
                enumerator_name = get_text(get_child(enumerator, "name"))
                members.append(Enum.Member(name=enumerator_name, value=value))
                self.add_enum_value(enumerator_name, value)
                if value is not None:
                    value += 1
            
            if typ_identifier is None: # It's an anonymous enum (possibly in a typedef, but renaming in that case will be handled down the stack in parse_typedef).
                typ = Enum(name=None, members=members)
            else:
                enum_name = get_text(typ_identifier)
                typ = Enum(name=enum_name, members=members)
                self.add_type("enum " + enum_name, typ)
        
        return typ
    
    def parse_function_signature(self, definition: Node) -> tuple[FunctionType, str]:
        """Obtain a function's type from its definition
        """
        assert definition.type == "function_definition", f"Scope.parse_function_signature accepts a function_definition AST node but a {definition.type} was passed."
        type_node = get_child(definition, "type")
        # We can set exapnded=True here because we except all types are fully defined before they are used in a function definition.
        base_type = self.get_type(get_text(type_node), expanded=True)
        if base_type is None:
            base_type = self.parse_type(type_node)
        if base_type is None:
            raise TypeNotFoundError(f"Return type {get_text(type_node)} can't be parsed in function definition.")
        # We can set exapnded=True here because we except all types are fully defined before they are used in a function definition.
        typ, name = self.parse_declarators(get_child(definition, "declarator"), base_type, expanded=True)
        assert isinstance(typ, FunctionType), f"A function definition should have a function type."
        if any(isinstance(p[0], Array) for p in typ.parameters):
            # Arrays in parameters in C decay to pointers, regardless of the size of the array.
            # This only occurs at the top level of pointer. So foo(int x[4]) means exactly the same thing as foo(int *).
            # We canonicalize the representation to the pointer notation.
            typ = FunctionType(typ.return_type, [
                (Pointer(ptype.element_type) if isinstance(ptype, Array) else ptype, pname) 
                for ptype, pname in typ.parameters
            ])
        self.add_function(typ, name)
        return typ, name
    
    @overload
    def parse_declaration(self, declaration: Node, compile: Literal[False], current_block: None = None, blocks = None) -> list[tuple[CType, str]]:
        ...

    @overload
    def parse_declaration(self, declaration: Node, compile: Literal[True], current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]] | None = None) -> list[tuple[CType, str]]:
        ...
    
    def parse_declaration(self, declaration: Node, compile: bool, current_block: BlockPointer | None = None, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]] | None = None) -> list[tuple[CType, str]]:
        """Process variable/functions declarations, adding them to this scope.

        If compile is False, then parse_declaration will return a list of the variable names and types declared.
        For instance, for "int x, *y;", [(PrimitiveType('int', 4), 'x'), (Pointer(PrimitiveType('int', 4)), 'y')] will be returned.
        If compile is true, then parse_declaration will output IR in the form of a sequence of VarInstructions.
        This will be empty unless there are any initializers.
        """
        assert not compile or current_block is not None, "A BlockPointer is required when compiling declarations."
        
        # statement.children[0]: (type) - the type of the parameter
        # statement.children[1]: (declarator) - the declarator
        # statement.children[2]: ; or ,
        # then repeated unnamed declarators and commas until the terminating ;.

        type_node = get_child(declaration, "type")
        # This is where variables' types are derived from, so it's important we set expanded=True to get the full type.
        base_type = self.get_type(get_text(type_node), expanded=True)
        if base_type is None:
            base_type = self.parse_type(type_node)
        if base_type is None:
            raise TypeNotFoundError(f"Declaration {get_text(declaration)} does not have a parsable type: {get_text(type_node)}")

        # if the declarator is an init_declarator,
        # declarator.children[0] (declarator) - the name of the variable being declared, or a declarator for it
        # declarator.children[1] =
        # declarator.children[2] (value) - the expression used to initialize the variable.

        output = [] # Either a list of declared variable types and names, or the instructions compiling this code would generate.
        for declarator in declaration.children_by_field_name("declarator"):
            if declarator.type == "init_declarator":
                value = declarator.child_by_field_name("value")
                declarator = get_child(declarator, "declarator")
            else:
                value = None

            # If we're compiling this declaration, that means we care about the code generated by the initializer (if on exists),
            # and so we need the full expanded type. Otherwise, we can defer the expansion to later.
            typ, name = self.parse_declarators(declarator, base_type, expanded=compile)
            # Some arrays are declared with no size and are sized based on the initializer's or string's length.
            if isinstance(typ, Array) and typ.nelements == 0 and value is not None:
                if value.type == "initializer_list":
                    assert value.children[0].type == "{" and value.children[-1].type == "}"
                    typ = Array(typ.element_type, (len(value.children) - 1) // 2)
                else:
                    assert value.type == "string_literal" or value.type == "concatenated_string"
                    string = check_expression_leaf(value, self)
                    assert isinstance(string, StringLiteral)
                    typ = string.type # CTypes are immutable so we can just make a copy here.
            if not compile:
                output.append((typ, name))
            if isinstance(typ, FunctionType):
                self.add_function(typ, name)
                continue
            else:
                new_variable = self.add_variable(typ, name)

            if compile and value is not None: # only true when there's an init_declarator
                assert current_block is not None and blocks is not None
                if value.type == "initializer_list":
                    partial_op_info = convert_initializer(value, self, typ, current_block, blocks)
                else:
                    partial_op_info = convert_instruction(value, self, current_block, blocks)
                opcode, operands, ast_node = partial_op_info
                current_block.block.instructions.append(VarInstruction(opcode, new_variable, operands, ast_node))

        return output # The variable was declared, but nothing was assigned to it. No computation was done.

    DECLARATOR_NODE_TYPES = {
        "identifier",
        "field_identifier",
        "type_identifier",
        "pointer_declarator",
        # "init_declarator", # handled specially in parse_declaration
        "array_declarator",
        "function_declarator",
        "parenthesized_declarator"
    }

    def parse_declarators(self, declarator: Node, base_type: CType, expanded: bool) -> tuple[CType, str]:
        assert declarator.type in Scope.DECLARATOR_NODE_TYPES, f"Unexpected declarator type: {declarator.type}: {get_text(declarator)}"

        # if declarator.type == "init_declarator":
        #     # declarator.children[0] (declarator) - the name of the variable being declared, or a declarator for it
        #     # declarator.children[1] =
        #     # declarator.children[2] (value) - the expression used to initialize the variable.
        #     declarator = get_child(declarator, "declarator")

        # Pointer declarators can be nested arbitrarily deep (e.g. int ****** x).
        if declarator.type == "pointer_declarator":
            # param_declarator.children[0] (None) is an *
            # param_declarator.children[1] (declarator) is another declarator - possibly a pointer.
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, Pointer(base_type), expanded)
        
        if declarator.type == "array_declarator":
            # declarator.children[0]: (declarator) - another declarator
            # declarator.children[1]: [
            # declarator.children[2]: (size; optional) - the array size
            # declarator.children[3]: ]
            # size = int(get_child(declarator, "size").text.decode())
            size = get_array_size(declarator)
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, Array(nelements=size, element_type=base_type), expanded)
        
        if declarator.type == "function_declarator":
            # declarator.children[0]: (declarator)
            # declarator.children[1]: (parameters)
            parameters = self.parse_parameters(get_child(declarator, "parameters"), expanded=expanded)
            declarator = get_child(declarator, "declarator")
            return self.parse_declarators(declarator, FunctionType(return_type=base_type, parameters=parameters), expanded)

        if declarator.type == "parenthesized_declarator":
            descendant = get_parenthesized_declarator(declarator)
            return self.parse_declarators(descendant, base_type, expanded)
        
        assert declarator.type == "field_identifier" or declarator.type == "type_identifier" or declarator.type == "identifier"
        return base_type, get_text(declarator)

    def parse_abstract_declarators(self, declarator: Node, base_type: CType, expanded: bool) -> tuple[CType, Optional[str]]:
        if declarator.type == "abstract_pointer_declarator":
            descendant = declarator.child_by_field_name("declarator")
            typ = Pointer(base_type)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ, expanded=expanded)
        elif declarator.type == "abstract_array_declarator":
            descendant = declarator.child_by_field_name("declarator")
            size = get_array_size(declarator)
            typ = Array(nelements=size, element_type=base_type)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ, expanded=expanded)
        elif declarator.type == "abstract_function_declarator":
            descendant = declarator.child_by_field_name("declarator")
            parameters = self.parse_parameters(get_child(declarator, "parameters"), expanded=expanded)
            typ = FunctionType(return_type=base_type, parameters=parameters)
            if descendant is None:
                return typ, None
            else:
                return self.parse_abstract_declarators(descendant, typ, expanded=expanded)
        elif declarator.type == "abstract_parenthesized_declarator":
            descendant = get_parenthesized_declarator(declarator)
            return self.parse_abstract_declarators(descendant, base_type, expanded=expanded)

        return self.parse_declarators(declarator, base_type, expanded=expanded)

    def parse_parameters(self, param_list: Node, expanded: bool) -> list[tuple[CType | FunctionType.VariadicParameter, Optional[str]]]:
        assert param_list.type == "parameter_list"
        assert param_list.children[0].type == "(" and param_list.children[-1].type == ")"
        parameters: list[tuple[CType | FunctionType.VariadicParameter, Optional[str]]] = []
        for param in param_list.children[1:-1]:
            if param.type == ",":
                continue
            if param.type == "variadic_parameter":  # variable number of arguments, denoted ...
                parameters.append((FunctionType.VariadicParameter(), None))
                continue
            # ANSI C parameters are of type parameter_declaration
            if param.type == "identifier":
                raise UnsupportedFeatureError(f"K&R-style parameter lists are not supported. (param = {get_text(param)})")
            typ_node = get_child(param, "type")
            base_type = self.get_type(get_text(typ_node), expanded=expanded)
            if base_type is None:
                base_type = self.parse_type(typ_node)
            if base_type is None:
                raise TypeNotFoundError(f"Parameter declaration with unknown type: {get_text(param)}: {get_text(typ_node)}")
            declarator = param.child_by_field_name("declarator")
            if declarator is None:
                parameters.append((base_type, None))
            else:
                parameters.append(self.parse_abstract_declarators(declarator, base_type, expanded=expanded))

        return parameters
    
    def __str__(self):
        components = ["TypeMapping:", "  Aliases:"]
        components.extend(
            f"    {alias}: {orig}"
            for alias, orig in self.aliases.items()
        )
        components.append("  Types:")
        components.extend(
            f"    {name}: {typ}"
            for name, typ in self.types.items()
        )
        components.append("  Incomplete Types:")
        components.extend(
            f"    {name}: {stub}"
            for name, stub in self.stubs.items()
        )
        components.append("  Enum Values:")
        components.extend(
            f"    {name}=" + ("<expr>" if value is None else str(value))
            for name, value in self.enum_values.items()
        )
        components.append("  Builtins:")
        components.extend(
            f"    {name}" for name in self.builtins
        )
        return "\n".join(components)


### Some utility functions for working the Node objects
def get_text(node: Node) -> str:
    """A wrapper around .text.decode() that fails with an exception when .text is None.
    """
    assert node.text is not None
    return node.text.decode('utf8')

def get_child(node: Node, child: int | str) -> Node:
    """A wrapper around .child_by_field_name and .children[] that fails with an exception
    where there is no such child.
    """
    if isinstance(child, str):
        ret = node.child_by_field_name(child)
        assert ret is not None, f"{node.type} has no child named {child}: {', '.join(f'{c.type}: {c.grammar_name}' for c in node.children)}"
    else:
        ret = node.children[child]
    return ret

def get_parenthesized_declarator(node: Node) -> Node:
    """Return the declarator inside a parenthesized declarator.

    Calling-convention modifiers such as ``__fastcall`` may appear between the
    opening parenthesis and the actual declarator, and tree-sitter does not
    always give the nested abstract declarator a field name in that case.
    """
    if declarator := node.child_by_field_name("declarator"):
        return declarator
    for child in node.children:
        if child.type.endswith("declarator"):
            return child
    raise AssertionError(
        f"{node.type} has no declarator child: "
        + ", ".join(f"{c.type}: {c.grammar_name}" for c in node.children)
    )

def get_array_size(declarator: Node) -> int:
    """Get the array size from an array_declarator or abstract_array_declarator
    """
    size_node = declarator.child_by_field_name("size")
    if size_node:
        if size_node.type == "number_literal":
            size = parse_int(get_text(size_node))
        else: # There's some kind of expression determining the size.
            size = -1
    else: # This is possible with a flexible array member in a struct. They have no inherent size; the extra space must be allocated dynamically.
        size = 0
    return size

def bits_to_signed_int(value: int, width: int) -> int:
    sign_bit = 1 << (width - 1)
    if value & sign_bit:
        return value - (1 << width)
    return value

def parse_int(s: str) -> int:
    """Return the numeric integer value that corresponds to the string.
    """
    s = s.lower()
    while s[-1] == "u" or s[-1] == "l":
        s = s[:-1]
    if s[:2] == "0x":
        return int(s, base=16)
    elif s[0] == "0":
        return int(s, base=8)
    else:
        return int(s)

def integer_type_max_value(type: Integer) -> int:
    if isinstance(type, SignedInteger):
        return (1 << (type.size * 8 - 1)) - 1
    assert isinstance(type, UnsignedInteger)
    return (1 << (type.size * 8)) - 1

def integer_literal_components(s: str) -> tuple[int, int, str, int | None]:
    """Return value, base, suffix, and the bit width implied by hex/octal spelling."""
    s = s.lower()
    if s.startswith("0x"):
        digit_start = 2
        base = 16
        while digit_start < len(s) and s[digit_start] in _HEX_DIGITS:
            digit_start += 1
        digits = s[2:digit_start]
        suffix = s[digit_start:]
        return int(digits, base), base, suffix, len(digits) * 4

    digit_end = 0
    while digit_end < len(s) and s[digit_end].isdigit():
        digit_end += 1
    digits = s[:digit_end]
    suffix = s[digit_end:]
    if s.startswith("0") and len(digits) > 1:
        octal_digits = digits[1:].lstrip("0")
        if octal_digits == "":
            digit_width = 1
        else:
            digit_width = (len(octal_digits) - 1) * 3 + int(octal_digits[0]).bit_length()
        return int(digits, 8), 8, suffix, digit_width
    return int(digits), 10, suffix, None

def integer_literal_type(raw_value: int, base: int, suffix: str) -> Integer:
    unsigned = "u" in suffix
    long_count = suffix.count("l")

    if unsigned:
        if long_count >= 2:
            return PRIMITIVE_TYPES["unsigned long long"]
        if long_count == 1:
            return PRIMITIVE_TYPES["unsigned long"]
        return PRIMITIVE_TYPES["unsigned int"]

    # A signed long/long long suffix is explicit enough for decompiler
    # bit-pattern literals that we keep the signed type even if the raw
    # non-decimal value would not fit as a mathematical C value.
    if long_count >= 2:
        return PRIMITIVE_TYPES["long long"]
    if long_count == 1:
        return PRIMITIVE_TYPES["long"]

    if base == 10:
        candidates = ("int", "long", "long long")
    else:
        candidates = ("int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long")
    for type_name in candidates:
        typ = PRIMITIVE_TYPES[type_name]
        assert isinstance(typ, Integer)
        if raw_value <= integer_type_max_value(typ):
            return typ
    return PRIMITIVE_TYPES["unsigned long long"]

def parse_integer_literal(s: str) -> IntegerConstant:
    if s.startswith(("-", "+")):
        literal = parse_integer_literal(s[1:])
        value = -literal.value if s[0] == "-" else literal.value
        return IntegerConstant(value, literal.type)

    raw_value, base, suffix, spelled_bit_width = integer_literal_components(s)
    typ = integer_literal_type(raw_value, base, suffix)
    value = raw_value
    if spelled_bit_width is not None and isinstance(typ, SignedInteger):
        type_width = typ.size * 8
        interpretation_width = min(type_width, max(INTEGER.size * 8, spelled_bit_width))
        value = bits_to_signed_int(raw_value % (1 << interpretation_width), interpretation_width)
    return IntegerConstant(value, typ)
    
def parse_float(s: str) -> float:
    """Return a float value that corresponds to the string.
    """
    s = s.lower()
    if s[-1] == "l" or s[-1] == "f":
        s = s[:-1]
    return float(s)

def remove_curly_braces(nodes: list[Node]) -> list[Node]:
    assert nodes[0].type == "{" and nodes[-1].type == "}"
    return nodes[1:-1]


def print_types_recursively(node: Node):
    print(node.type)
    for child in node.children:
        print_types_recursively(child)

def print_immediate_children(root: Node):
    for i, child in enumerate(root.children):
        print(child.type, end=": ")
        print(root.field_name_for_child(i))

_SIMPLE_ESCAPE_VALUES = {
    "'": ord("'"),
    '"': ord('"'),
    "?": ord("?"),
    "\\": ord("\\"),
    "a": 7,
    "b": 8,
    "f": 12,
    "n": 10,
    "r": 13,
    "t": 9,
    "v": 11,
}

_HEX_DIGITS = "0123456789abcdefABCDEF"
_UCN_BASIC_CHARACTER_EXCEPTIONS = {0x24, 0x40, 0x60}

def integer_max_value(type: Integer) -> int:
    return (1 << (type.size * 8)) - 1

def validate_universal_character_name(codepoint: int, value: str):
    if codepoint > 0x10ffff or 0xd800 <= codepoint <= 0xdfff:
        raise ParsingError(f"Invalid universal character name {value!r}.")
    if codepoint < 0x00a0 and codepoint not in _UCN_BASIC_CHARACTER_EXCEPTIONS:
        raise ParsingError(f"Invalid universal character name {value!r}.")

def escape_sequence_value(value: str) -> int:
    escape = value[1]
    if escape in _SIMPLE_ESCAPE_VALUES:
        if len(value) != 2:
            raise SemanticError(f"Character literal {value!r} does not fit in one {CHARACTER}.")
        return _SIMPLE_ESCAPE_VALUES[escape]

    if escape in "01234567":
        digits = value[1:]
        if len(digits) > 3 or any(digit not in "01234567" for digit in digits):
            raise SemanticError(f"Character literal {value!r} does not fit in one {CHARACTER}.")
        return int(digits, 8)

    if escape == "x":
        digits = value[2:]
        if len(digits) == 0 or any(digit not in _HEX_DIGITS for digit in digits):
            raise ParsingError(f"Invalid hex escape sequence {value!r}.")
        return int(digits, 16)

    if escape in ("u", "U"):
        digits = value[2:]
        digit_count = 4 if escape == "u" else 8
        if len(digits) != digit_count or any(digit not in _HEX_DIGITS for digit in digits):
            raise ParsingError(f"Invalid universal character name {value!r}.")
        codepoint = int(digits, 16)
        validate_universal_character_name(codepoint, value)
        return codepoint

    raise ParsingError(f"Invalid escape sequence {value!r}.")

def character_value(value: str) -> int:
    if len(value) == 1:
        return ord(value)
    if not value.startswith("\\") or len(value) == 1:
        raise SemanticError(f"Character literal {value!r} does not fit in one {CHARACTER}.")
    return escape_sequence_value(value)

def validate_character_fits(value: str):
    if character_value(value) > integer_max_value(CHARACTER):
        raise SemanticError(f"Character literal {value!r} does not fit in one {CHARACTER}.")

def parse_escape_sequence(value: str, start: int) -> tuple[str, int]:
    i = start + 1
    if i == len(value):
        raise ParsingError("Invalid escape sequence '\\'.")

    if value[i] in "01234567":
        i += 1
        while i < len(value) and i - start < 4 and value[i] in "01234567":
            i += 1
    elif value[i] == "x":
        i += 1
        if i == len(value) or value[i] not in _HEX_DIGITS:
            raise ParsingError(f"Invalid hex escape sequence {value[start:i]!r}.")
        while i < len(value) and value[i] in _HEX_DIGITS:
            i += 1
    elif value[i] == "u" or value[i] == "U":
        i += (5 if value[i] == "u" else 9)
        raise UnsupportedFeatureError(f"Universal character names unsupported: {value[start:i]}.")
    else:
        i += 1

    return value[start:i], i

def parse_character_literal(value: str, type: SignedInteger = INTEGER) -> CharLiteral:
    validate_character_fits(value)
    return CharLiteral(character_value(value), type)

def parse_string_literal_content(value: str) -> list[CharLiteral]:
    chars = []
    i = 0
    while i < len(value):
        if value[i] != "\\":
            char = value[i]
            validate_character_fits(char)
            chars.append(CharLiteral(character_value(char), CHARACTER))
            i += 1
            continue

        char, i = parse_escape_sequence(value, i)
        validate_character_fits(char)
        chars.append(CharLiteral(character_value(char), CHARACTER))

    chars.append(CharLiteral(0, CHARACTER))
    return chars

def check_expression_leaf(expression: Node, scope: Scope) -> Variable | Constant | None:
    if expression.type == "identifier":
        text = get_text(expression)
        # # TODO: This while loop should probably be refactored into scope directly to better model shadowing
        # # and catch name conflicts. We must at least have it here to distinguish between enum constants
        # # and retular Variables.
        # search = scope
        # while search is not None:
        #     if text in search.variables:
        #         break
        #     elif text in search.enum_values:
        #         value = search.enum_values[text]
        #         if value is None:
        #             raise SemanticError(f"Enum constant {text} has no value.")
        #         return IntegerConstant(value, INTEGER)
        #     search = search.enclosing_scope
        return scope.check_variable(text) # will add the variable.
    if expression.type == "number_literal":
        number_text = get_text(expression).lower()
        if "." in number_text: # is a float literal
            # Hexidecimal floating point numbers are allowed in C too but we don't support those for now.
            type_name = "long double" if "l" in number_text else ("float" if "f" in number_text else "double")
            return FloatConstant(parse_float(number_text), PRIMITIVE_TYPES[type_name])
        else: # is an integer literal
            return parse_integer_literal(number_text)
    if expression.type == "string_literal":
        # string literals, as parsed by tree-sitter, can contain any number of string_content and escape_sequence nodes
        # in alternating order. However, because the first and last nodes are always just ", we just ignore them and get
        # the rest of the content of the string.
        if expression.children[0].type != '"' or expression.children[-1].type != '"':
            raise ParsingError(f"Only ordinary C99 string literals are supported: {get_text(expression)}")
        literal_text = get_text(expression)[1:-1]
        return StringLiteral(literal_text, parse_string_literal_content(literal_text))
    if expression.type == "char_literal":
        if expression.children[0].type != "'" or expression.children[-1].type != "'":
            raise ParsingError(f"Only ordinary C99 character literals are supported: {get_text(expression)}")
        if len(expression.children) != 3:
            raise SemanticError(f"Character literal does not fit in one CHARACTER: {get_text(expression)}")
        character_node = expression.children[1]
        assert character_node.type == "character" or character_node.type == "escape_sequence", f"Character literal node type not recognized: {character_node.type}"
        return parse_character_literal(get_text(character_node))
    # true, TRUE, false, FALSE, and NULL are not keywords in C but tree-sitter recognizes them with their own node types anyway.
    if expression.type == "null":
        return IntegerConstant(0, SIZE_T) # platform dependent
    if expression.type == "true":
        return IntegerConstant(1, INTEGER)
    if expression.type == "false":
        return IntegerConstant(0, INTEGER)
    if expression.type == "concatenated_string":
        assert all(child.type == "string_literal" for child in expression.children)
        text = [get_text(child) for child in expression.children]
        if not all(t[0] == '"' and t[-1] == '"' for t in text):
            raise ParsingError(f"Only ordinary C99 string literals are supported: {get_text(expression)}")
        characters = []
        for i in range(len(text)):
            text[i] = text[i][1:-1]
            characters.extend(parse_string_literal_content(text[i])[:-1]) # :-1 to remove the null char
        characters.append(CharLiteral(0, CHARACTER))
        return StringLiteral("".join(text), characters)
    
    return None

def clean_expression(expression: Node) -> Node:
    while expression.type == "parenthesized_expression":
        assert(expression.children[0].type == "(")
        assert(expression.children[-1].type ==")")
        assert(len(expression.children) == 3)
        expression = expression.children[1]
    return expression

ASSIGNMENT_SUBOPS = {
    "+=": Addition(),
    "-=": Subtraction(),
    "*=": Multiplication(),
    "/=": Division(),
    "%=": ModulusDivision(),
    "<<=": LeftShift(),
    ">>=": RightShift(),
    "&=": BitwiseAnd(),
    "^=": BitwiseXOr(),
    "|=": BitwiseOr()
}

UNARY_OPS = { # Excludes pointer * and & because these appear as part of pointer_expression nodes
              # rather than unary_expression nodes.
    "-": UnaryMinus(),
    "!": LogicalNot(),
    "~": BitwiseNot(),
}

INFIX_OPS = {
    "+": Addition(),
    "-": Subtraction(),
    "*": Multiplication(),
    "/": Division(),
    "%": ModulusDivision(),
    "<<": LeftShift(),
    ">>": RightShift(),
    "<": LessThan(),
    "<=": LessThanOrEqualTo(),
    ">": GreaterThan(),
    ">=": GreaterThanOrEqualTo(),
    "==": EqualTo(),
    "!=": NotEqualTo(),
    "&": BitwiseAnd(),
    "|": BitwiseOr(),
    "^": BitwiseXOr(),
    "&&": LogicalAnd(),
    "||": LogicalOr()
}

def get_assignment_subopcode(instruction: Node) -> Operation:
    assert instruction.type != "=", "'=' does not have a sub-opcode; other assignment instructions do (e.g. +=)."
    assert instruction.type in ASSIGNMENT_SUBOPS, f"{instruction.type} not a valid C assignment instruction."
    return ASSIGNMENT_SUBOPS[instruction.type]

# C expressions are recursively defined entities. The IR requres expressions be represented as a sequence
# of single operations, with the result of each stored in a variable. A set of three functions, which form
# a three-way recursive system, help perform this conversion. The recursion can be started from any of the
# three functions, depending on what is desired (see the end of this comment for details).
#
# The relationships between the functions are as follows, where -> indicates a call.
#  ... -> bind_expression -> convert_instruction -> expand_subexpression -> ...
#
########## convert_instruction ##########
# convert_instruction converts a tree-sitter AST node into partial components for an instruction: it extracts the 
# opcode and the arguments but does not create a VarInstruction IR object. Rather, it returns these components.
# This is because an IR VarInstruction requires one additional piece of information: where to store the result
# of the computation. Determining this is the responsibility of the bind_expression function.
#
########## bind_expression ##########
# The role of the bind_expression function is to determine in what variable the results of a given instruction
# should be stored. Determining what the instruction and its arguments actually are is the responsibility of 
# convert_instruction; bind_expression calls convert_instruction for this purpose. bind_expression stores 
# the result of an instruction in a variable from the original program if possible (e.g. in "int a = b + c;",
# the result will be stored in "a"). Sometimes, however, the instruction's result is, in the original program,
# used as part of a larger expression (as "4 * b" is in "int a = 4 * b + 1;"). In this case, bind_expression
# will assign the operation's result to a temporary variable.
#
########## expand_subexpression ##########
# expand_subexpression is a wrapper around bind_expression that helps prevent unnecessary "copy" operations.
# A "copy" operation stores the value of a variable or constant into a variable, as in "x = 3;" or "y = x;"
# If expand_subexpression encounters a "leaf" node, (a variable or constant), it simply returns it. Otherwise,
# it forwards the expression to bind_expression. bind_expression always binds the expression it recieves to 
# a variable, creating a fresh temporary if necessary. This is undesirable when the expression is simply 
# a variable or constant itself used as an argument to another instruction. For example, if calls were routed
# directly to bind_expression, "x = y + 2;" would be converted into
#   t0 = y
#   t1 = 2
#   x = + t0 t1
# which has two redundant copy instructions.
#
# 
#
########## initiating recursion ##########
# There are situations where calling each of these functions to initiate expression parsing makes sense.
# Call bind_expression if
# - The top-level expression can be converted to an instruction no matter what. (This is handy for genuine 
#   copy operations present in the original code like x = y;).
# Call expand_subexpression if
# - You want a value (variable or constant) that is the result of the entire expression.
# Call convert_instruction if
# - You want to bind the operation and arguments to a variable yourself.

def bind_expression(expression: Node, scope: Scope, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> Variable:
    operands: list[VarOperand] # Just declare this here for the typechecker's sake.
    expression = clean_expression(expression)

    if expression.type == "assignment_expression":
        # expression.children[0]: (left) - the lhs of the assignment
        # expression.children[1]: (operator) - either = or +=
        # expression.children[2]: (right) - the rhs of the assignment

        ## Process the LHS
        lhs = clean_expression(get_child(expression, "left"))
        if (lhs.type == "identifier"):
            result_var = scope.check_variable(get_text(lhs))
            store_required = False
        else:
            assert lhs.type != "assignment_expression", "Assignment expressions should be on the rhs"
            # Handle the case where the LHS is an expression, as in 'point->x'.
            result_var = bind_expression(lhs, scope, current_block, blocks)
            store_required = True
        
        ## Process the RHS
        rhs = get_child(expression, "right")
        assignment_instruction = get_child(expression, "operator")
        if store_required:
            # Consider the expression pt->x = pt->y = var. This will decompose into:
            #   t0 = pt->x
            #   t1 = pt->y
            #   t1 = store t1 var
            #   t0 = store t0 t1
            # Here, t0 and t1 represent subobjects of pt. We may end up treating a subobject as both an
            # lval (memory address: where to store the result) and an rval (the value being stored).
            # Note that the store instruction returns the subobject at which the value was stored, not the 
            # original value stored. This is relevant if the act of storing the value modifies the value
            # (e.g. storing a float in an int variable.)
            # 
            # In the first store instruction, the first argument corresponds to the subobject for the y field of pt.
            # It's used as the location to store the value contained in "var." Meanwhile, the second argument of the 
            # second store instance takes the same suboject but reads its value. The left operand is always treated
            # as an lval, and the right as an rval.
            value_to_store = expand_subexpression(rhs, scope, current_block, blocks)
            if assignment_instruction.type != "=": # could be +=, -=, etc.
                subopcode = get_assignment_subopcode(assignment_instruction)
                # This also commits the subtle type mismatch described above: result_var is treated as both 
                # an address and a value. (It is treated as a value in this operand and in the lhs processing
                # code but as an address in the STORE_OP).
                temporary_variable = scope.create_temporary()
                current_block.block.instructions.append(VarInstruction(subopcode, temporary_variable, [result_var, value_to_store], ast_node=expression))
                value_to_store = temporary_variable # so the correct value_to_store is included in the arguments to the store instruction being constructed.
            opcode = STORE_OP # arguments (address, value to store)
            operands = [result_var, value_to_store]
            ast_node = expression
        else:
            if assignment_instruction.type == "=":
                if rhs.type == "assignment_expression" or rhs.type == "update_expression":
                    # Here, we have a nested assignment expression, e.g. a = b = 1;
                    operand = bind_expression(rhs, scope, current_block, blocks)
                    # If the right hand side is an assignment expression, then it will bind the result
                    # to some variable (returned as operand just above). In the case of a = b = 1, that variable
                    # will be "b". The only thing left to do is copy the result from that variable into the one on 
                    # the lhs of this assignment (in the example, "a").
                    opcode = COPY_OP
                    operands = [operand]
                    ast_node = None
                else: # The rhs is not an assignment expression.
                    # This is the "typical" case for an assignment expression: the lhs is simply a variable, and the 
                    # right hand side is a nonassignment expression. The assignment expression uses a plain =.
                    # Examples include x = x + 1; and a = foo(b, c);
                    opcode, operands, ast_node = convert_instruction(rhs, scope, current_block, blocks)
            else: # the assignment is a +=, -=, etc.
                # Get a single variable or constant representing the rhs of the expression, then build the appropriate binary
                # instruction operating on the lhs variable and the new rhs variable.
                rhs_result = expand_subexpression(rhs, scope, current_block, blocks)
                # set up variables so that this instruction can be built correctly.
                opcode = get_assignment_subopcode(assignment_instruction)
                operands = [result_var, rhs_result]
                ast_node = expression
    elif expression.type == "update_expression":
        # ++ and --
        # The children of expression are 'argument' and 'operator'; the order depends on if prefix or postifix form is used.
        operand = expand_subexpression(get_child(expression, "argument"), scope, current_block, blocks)
        opcode = get_text(get_child(expression, "operator"))
        assert opcode == "++" or opcode == "--"
        opcode = Addition() if opcode == "++" else Subtraction()

        # Someone could write something like 2++. This does not compile in C and is generally nonsensical.
        if not isinstance(operand, Variable):
            raise SemanticError(f"Cannot apply update instruction {opcode} to expression \"{operand}\" of type \"{type(operand)}\"")
        
        if expression.field_name_for_child(0) == "operator": # is prefix (++i). Easy case.
            operands = [operand, ONE]
            result_var = operand
            ast_node = expression
        else: # is postfix (i++).
            # To handle postfix update expressions (e.g. i++), we decompose them into two instructions: a copy instruction
            # and a binary arithmetic instruction (+ or -). Importantly, the return value (the variable representing the
            # overall expression) is the temporary variable, not variable being updated. This is because if i++ is used
            # in an expression, that use should reflect the unincremented version of value of i, as is the semantics of
            # the postfix update expression. The copy instruction will eventually be eliminated as part of copy propagation
            # when converting to SSA.

            # Note that we can't use the instruction-building code below because the final instruction we want to create is the
            # increment/decrement operation, but we want to return the value from the copy operation that reflects the value
            # of operand before the increment/decrement.
            result_var = scope.create_temporary()
            current_block.block.instructions.append(VarInstruction(COPY_OP, result_var, [operand], None)) # must come first to reflect the value of operand before the update
            current_block.block.instructions.append(VarInstruction(opcode, operand, [operand, ONE], expression))
            return result_var
    # Short-circuiting operations necessarily must already have a bound variable (which holds the result of each branch).
    elif scope.short_circuit_logical_ops and expression.type == "binary_expression" and isinstance(logical_operation := INFIX_OPS[get_text(get_child(expression, "operator"))], (LogicalAnd, LogicalOr)):
        return bind_short_circuit_logical(logical_operation, expression, scope, current_block, blocks)
    elif expression.type == "conditional_expression":
        return bind_conditional_expression(expression, scope, current_block, blocks)
    else: # Is not an assignment expression. Bind to a temporary variable
        opcode, operands, ast_node = convert_instruction(expression, scope, current_block, blocks)
        result_var = scope.create_temporary()
    
    # Reaching this point requires: result_var, opcode, operands, and ast_node from the above if/else
    current_block.block.instructions.append(VarInstruction(opcode, result_var, operands, ast_node=ast_node))

    return result_var

def bind_short_circuit_logical(opcode: LogicalAnd | LogicalOr, expression: Node, scope: Scope, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> Variable:
    assert expression.type == "binary_expression"
    
    left_operand = expand_subexpression(get_child(expression, "left"), scope, current_block, blocks)
    if_block = current_block.block
    if_block.instructions.append(VarInstruction(IF_OP, None, [left_operand], ast_node=expression))

    rhs_block = BasicBlock([], [], [])
    short_circuit_block = BasicBlock([], [], [])
    merge_block = BasicBlock([], [], [])

    blocks.append((if_block, None))

    result_var: Variable = scope.create_temporary(is_stack_allocated=True)
    result_var.type = INTEGER
    if isinstance(opcode, LogicalAnd):
        if_block.add_successor(rhs_block) # if true, then we evaluate the other argument. Thus we make this the first successor.
        if_block.add_successor(short_circuit_block) # if the first argument of an && operator is False, then short-circuit.

        short_circuit_block.instructions.append(VarInstruction(COPY_OP, result_var, [ZERO], ast_node=expression))
        short_circuit_block.add_successor(merge_block)
    else:
        assert isinstance(opcode, LogicalOr), f"Only LogicalAnd and LogicalOr are short-circuiting but found {opcode}"
        if_block.add_successor(short_circuit_block) 
        if_block.add_successor(rhs_block)

        short_circuit_block.instructions.append(VarInstruction(COPY_OP, result_var, [ONE], ast_node=expression))
        short_circuit_block.add_successor(merge_block)
    
    # Short circuit block is complete, so we can append it to the block list.
    blocks.append((short_circuit_block, None))

    rhs_ptr = BlockPointer(rhs_block)
    operation, operands, node = convert_instruction(get_child(expression, "right"), scope, rhs_ptr, blocks)
    # The C standard specifies that the output of && and || will be either 0 or 1. If the rhs ends in a conditional
    # expression like x < 4, then this is already the case. However, if it is just a value (e.g. if (a && b)), then 
    # this is not the case, since b is could be any value. Here, we do such conversions if necessary.
    if isinstance(operation, (LessThan, LessThanOrEqualTo, GreaterThan, GreaterThanOrEqualTo, EqualTo, NotEqualTo)):
        rhs_ptr.block.instructions.append(VarInstruction(operation, result_var, operands, ast_node=node))
    else:
        # If it's a copy operation, we don't need to assign the result to a temporary and then put that in the !=; we can put it in the != directly.
        if isinstance(operation, Copy):
            assert len(operands) == 1
            rhs_block_result = operands[0]
        else:
            rhs_block_result = scope.create_temporary()
            rhs_ptr.block.instructions.append(VarInstruction(operation, rhs_block_result, operands, ast_node=node))
        rhs_ptr.block.instructions.append(VarInstruction(INFIX_OPS["!="], result_var, [rhs_block_result, ZERO]))
    rhs_ptr.block.add_successor(merge_block)
    # Rhs block is complete; add it to the block list.
    blocks.append((rhs_ptr.block, None))

    current_block.block = merge_block
    return result_var

def bind_conditional_expression(expression: Node, scope: Scope, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> Variable:
    assert expression.type == "conditional_expression"
    condition_node = get_child(expression, "condition")
    consequence_node = get_child(expression, "consequence")
    alternative_node = get_child(expression, "alternative")

    condition = expand_subexpression(condition_node, scope, current_block, blocks)
    if_block = current_block.block
    blocks.append((if_block, None))

    consequence_block = BasicBlock([], [], [])
    alternative_block = BasicBlock([], [], [])
    merge_block = BasicBlock([], [], [])
    blocks.extend([(consequence_block, None), (alternative_block, None)])

    if_block.instructions.append(VarInstruction(IF_OP, None, [condition], expression))
    if_block.add_successor(consequence_block)
    if_block.add_successor(alternative_block)

    result_var = scope.create_temporary(is_stack_allocated=True)

    cons_holder = BlockPointer(consequence_block)
    operation, operands, node = convert_instruction(consequence_node, scope, cons_holder, blocks)
    cons_holder.block.instructions.append(VarInstruction(operation, result_var, operands, node))
    cons_holder.block.add_successor(merge_block)

    alt_holder = BlockPointer(alternative_block)
    operation, operands, node = convert_instruction(alternative_node, scope, alt_holder, blocks)
    alt_holder.block.instructions.append(VarInstruction(operation, result_var, operands, node))
    alt_holder.block.add_successor(merge_block)

    current_block.block = merge_block
    return result_var

def expand_subexpression(expression: Node, scope: Scope, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> Variable | Constant:
    expression = clean_expression(expression)
    operand = check_expression_leaf(expression, scope)
    if operand is None:
        operand = bind_expression(expression, scope, current_block, blocks)
    return operand

def convert_instruction(expression: Node, scope: Scope, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> tuple[Operation, list[VarOperand], Node | None]:
    assert expression.type != "assignment_expression" # Should be handled by bind_expression
    assert expression.type != "update_expression" # Should be handled by bind_expression
    expression = clean_expression(expression)

    leaf = check_expression_leaf(expression, scope)

    if leaf:
        # A copy instruction, e.g. int x = y; or int x = 3;
        # This should ONLY be reached if there's a direct copy statement like this that occurs in the code.
        # We do not want copy instructions to occur as part of a subexpression, e.g. x = y + z; should not
        # be converted into x = t1; t1 = y + z;
        return (COPY_OP, [leaf], None)
    elif expression.type == "unary_expression":
        # expression.children[0]: (operator) - the operation being performed (e.g. !)
        # expression.children[1]: (argument) - the operand

        operand = expand_subexpression(get_child(expression, "argument"), scope, current_block, blocks)
        
        opcode = UNARY_OPS[get_text(get_child(expression, "operator"))]
        return (opcode, [operand], expression)
    elif expression.type == "binary_expression":
        # expression.children[0]: (left) - the left operand
        # expression.children[1]: (operator) - the operation being performed (e.g. +, -)
        # expression.children[2]: (right) - the right operand
        opcode = INFIX_OPS[get_text(get_child(expression, "operator"))]
        if scope.short_circuit_logical_ops and isinstance(opcode, (LogicalAnd, LogicalOr)):
            result_var = bind_short_circuit_logical(opcode, expression, scope, current_block, blocks)
            return (COPY_OP, [result_var], None)

        left_operand = expand_subexpression(get_child(expression, "left"), scope, current_block, blocks)
        right_operand = expand_subexpression(get_child(expression, "right"), scope, current_block, blocks)
        return (opcode, [left_operand, right_operand], expression)
    elif expression.type == "call_expression":
        # expression.children[0]: (function) - the name of the function.
        # expression.children[1]: (arguments; argument_list) - a list of arguments.
        ftype = None # Will be populated if possible below

        # Get the name of the function
        name_node = get_child(expression, "function")
        if name_node.type == "identifier":
            # If we call "expand_subexpression" with an identifier, it will interpret that identifier as a
            # variable, which is incorrect. Instead, we manually extract the function name here.
            function_name = get_text(name_node) # function_name has type String

            # If the name of this function is an already defined variable. If it is, then that variable must
            # be a function call with a function pointer. If not, then we assume it is a standard call to a 
            # function with that name. (In theory, it could be a function call using a yet enencountered global variable,
            # but we don't have enough information to differentiate this case from the much more common normal 
            # function call).
            if scope.variable_exists(function_name):
                function_name = scope.check_variable(function_name)
                # function_name now has type Variable
            else:
                ftype = scope.get_function(function_name) # May return None if not found, which is fine.
        else:
            # This could be the (*function_pointer)(args) syntax. If so, we extract the function pointer variable name.
            if name_node.type == "parenthesized_expression" and  name_node.children[1].type == "pointer_expression":
                assert name_node.children[1].children[0].type == "*"
                name_node = name_node.children[1].children[1]
                # It is possible for name_node to be an identifier here. However, if the function is in (*function_pointer)(args)
                # syntax, we assume that this function is a function pointer variable (possibly a global one) because 
                # this syntax is predominantly used for function pointers, unlike the normal function-call syntax.
            
            # Regardless of which syntax is used, the resulting expression could arbitrarily complex. We therefore must call
            # expand_subexpression to handle it.
            function_name = expand_subexpression(name_node, scope, current_block, blocks)
            # If the function's name is a string, it should be detected under "if name_node.type == identifier" above.
            assert not isinstance(function_name, Constant), f"A function's name should be a string or a variable."

        # Process the arguments of the function
        arguments = get_child(expression, "arguments")
        assert(arguments.children[0].type == "(")
        assert(arguments.children[-1].type == ")")
        arguments = arguments.children[1:-1]

        operands = [] # the operands in this expression
        for argument in arguments:
            if argument.type == ",":
                continue
            operands.append(expand_subexpression(argument, scope, current_block, blocks))
        
        return (FunctionCall(function_name, ftype), operands, expression)
    elif expression.type == "pointer_expression":
        # expression.children[0]: (operator)
        # expression.children[1]: (argument) - the thing being dereferenced.
        operand = expand_subexpression(get_child(expression, "argument"), scope, current_block, blocks)
        operator_text = get_text(get_child(expression, "operator"))
        if operator_text == "&":
            operator = AddressOf()
        else:
            assert operator_text == "*"
            operator = Dereference()
        return (operator, [operand], expression)
    elif expression.type == "conditional_expression":
        # expression.children[0]: (condition) - the conditional part of the ternary
        # expression.children[1]: ?
        # expression.children[2]: (consequence) - the 'true' part of the ternary
        # expression.children[3]: :
        # expression.children[4]: (alternative) - the 'false' part of the ternary
        result_var = bind_conditional_expression(expression, scope, current_block, blocks)
        return (COPY_OP, [result_var], expression) # behaves like an phi instruction, but "implemented" with "real" instructions (the same variable is written to at the end of each branch.)
    elif expression.type == "field_expression":
        # expression.children[0]: (argument) - an expression that resolves to the struct
        # expression.children[1]: (operator) ->
        # expression.children[2]: (field) - the field being accessed.

        argument = expand_subexpression(get_child(expression, "argument"), scope, current_block, blocks)
        operator_text = get_text(get_child(expression, "operator"))
        assert operator_text == "." or operator_text == "->"
        field = Field(get_text(get_child(expression, "field")))

        return (MemberAccess(operator_text == "->"), [argument, field], expression)
    elif expression.type == "cast_expression":
        # expression.children[0]: (
        # expression.children[1]: (type) - the type being casted to
        # expression.children[2]: )
        # expression.children[3]: (value) - the expression to cast

        type_node = get_child(expression, "type")
        cast_type = scope.get_or_parse_from_type_descriptor(type_node, expanded=True)
        value = expand_subexpression(get_child(expression, "value"), scope, current_block, blocks)

        return (CAST_OP, [cast_type, value], expression)
    elif expression.type == "subscript_expression":
        # expression.children[0]: (argument) - the name of the array, or an expression that resolves to an array.
        # expression.children[1]: [
        # expression.children[2]: (index) - an expression that resolves to an array index.
        # expression.children[3]: ]

        array = expand_subexpression(get_child(expression, "argument"), scope, current_block, blocks)
        index = expand_subexpression(get_child(expression, "index"), scope, current_block, blocks)

        return (SUBSCRIPT_OP, [array, index], expression)
    elif expression.type == "sizeof_expression":
        # sizeof can take a type or an expression as an argument. We handle each case separately.

        # if this sizeof is describing a type:
        # expressions.children[0]: sizeof
        # expressions.children[1]: (
        # expressions.children[2]: (type) - the type for which the size is being measured.
        # expressions.children[3]: )
        type_descriptior = expression.child_by_field_name("type")
        if type_descriptior is not None: # It is sizeof(type)
            arg_type = scope.get_or_parse_from_type_descriptor(type_descriptior)
            return (SIZEOF_OP, [arg_type], expression)
        else:
            # expression.children[0]: sizeof
            # expression.children[1]: expression
            argument = expand_subexpression(expression.children[1], scope, current_block, blocks)
            return (SIZEOF_OP, [argument], expression)
    elif expression.type == "comma_expression":
        # expressions.children[0]: (left) - the first expression evaluated in the comma expression
        # expressions.children[1]: ,
        # expressions.children[2]: (right) - the second expression evaluated in the comma expression

        # If irrelevant values need not be bound to temporaries, then this can be refactored.
        _ = bind_expression(get_child(expression, "left"), scope, current_block, blocks) # left expression
        # We want this order because the left expression in a comma expression is evaluated first
        right = get_child(expression, "right")
        if right.type == "assignment_expression" or right.type == "update_expression":
            # This is a corner case where the right expression in a comma expression is itself an assignment 
            # expression, i.e. x = (expr(), y = expr2());. In this case, x will copy y.
            # This reduces the comma expression to a copy operation, copying the result of the right expression.
            right_result = bind_expression(right, scope, current_block, blocks)
            return (COPY_OP, [right_result], None)
        else:
            return convert_instruction(right, scope, current_block, blocks)
    elif expression.type == "compound_literal_expression":
        # Form is (type) initializer_list.
        typ = scope.get_or_parse_from_type_descriptor(get_child(expression, "type"), expanded=True)
        return convert_initializer(get_child(expression, "value"), scope, typ, current_block, blocks)
    elif expression.type == "ERROR":
        raise ParsingError(get_text(expression)) 
    else:
        raise NotImplementedError(f"No code yet implemented to handle expressions of type '{expression.type}'. ({get_text(expression)})")

def split_initializer_pair(pair: Node) -> tuple[str, Node]:
    assert pair.type == "initializer_pair"
    designator = get_child(pair, "designator")
    assert designator.children[1].type == "field_identifier"
    field_name = get_text(designator.children[1])
    value = get_child(pair, "value")
    return field_name, value

def bind_nested_initializer(node: Node, scope: Scope, nested_type: CType, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> Variable:
    """Assigns the result of an initializer to a temporary variable. Simlar to bind_expression,
    but for initializers because initializers are not expressions.
    """
    init_op, init_vals, ast_node = convert_initializer(node, scope, nested_type, current_block, blocks)
    temporary: Variable = scope.create_temporary()
    temporary.type = nested_type # may be unnecessary as we can just infer this in type inference later.
    current_block.block.instructions.append(VarInstruction(init_op, temporary, init_vals, ast_node))
    return temporary

def convert_initializer(node: Node, scope: Scope, init_t: CType, current_block: BlockPointer, blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]]) -> tuple[Initializer, list[VarOperand], Node | None]:
    assert node.type == "initializer_list", f"Expected initializer list but found {node}"
    assert node.children[0].type == "{" and node.children[-1].type == "}"
    field_names: list[str] | None
    element_iter = itertools.islice(node.children[1:-1], 0, None, 2) # Skip the 
    if isinstance(init_t, (Struct, Union)):
        name2index = {m.name: i for i, m in enumerate(init_t.members)}
        initial_values = []
        field_names = []
        current_index = 0
        for element in element_iter:
            if element.type == "initializer_pair":
                field_name, value = split_initializer_pair(element)
                current_index = name2index[field_name] + 1
            else:
                field_name = init_t.members[current_index].name
                value = element
                current_index += 1
            field_names.append(field_name)
            if value.type == "initializer_list":
                field_type = init_t.typeof(field_name)
                if field_type is None:
                    raise SemanticError(f"Error in compiling initalizer: {init_t} has no field named {field_name}")
                initial_values.append(bind_nested_initializer(value, scope, field_type, current_block, blocks))
            else:
                initial_values.append(expand_subexpression(value, scope, current_block, blocks))
    elif isinstance(init_t, (IncompleteStruct, IncompleteUnion)):
        initial_values = []
        field_names = []
        for element in element_iter:
            if element.type != "initializer_pair":
                raise UnsupportedFeatureError(f"Order-based initialization for incomplete structs and unions is not supported.")
            field_name, value = split_initializer_pair(element)
            field_names.append(field_name)
            if value.type == "initializer_list": # can't be processed because we don't know the type of the field being initialized.
                raise UnsupportedFeatureError(f"Cannot process a nested initializer list for an incomplete type.")
            initial_values.append(expand_subexpression(value, scope, current_block, blocks))
    else:
        initial_values = []
        for element in element_iter:
            if element.type == "initializer_pair":
                raise SemanticError(f"Attempting to initialize a {init_t} with initializer pair {get_text(element)}.")
            if element.type == "initializer_list":
                if isinstance(init_t, Array):
                    initial_values.append(bind_nested_initializer(element, scope, init_t.element_type, current_block, blocks))
                else:
                    raise SemanticError(f"Recursive initializer list for scalar type {init_t}: {get_text(node)}")
            else:
                initial_values.append(expand_subexpression(element, scope, current_block, blocks))
        field_names = None
    
    return Initializer(init_t, field_names), initial_values, node






#######################################################################################################
# Compile statments. Handle most control flow (except control-flow inducing expressions, done above.) #
#######################################################################################################

def convert_compound_statement(body: Node | list[Node], scope: Scope) -> list[tuple[BasicBlock, list[BlockSuccessorAssignment]]]:
    if isinstance(body, list):
        statements = body
    elif body.type == "compound_statement":
        assert body.children[0].type == "{"
        assert body.children[-1].type == "}"
        statements = body.children[1:-1]
    else:
        statements = [body] # body will be an individual statement; we must wrap it in a list to use the code below.

    blocks = []
    current_block = BasicBlock([], [], [])
    for statement in statements:
        if statement.type == "declaration":
            holder = BlockPointer(current_block)
            scope.parse_declaration(statement, True, holder, blocks)
            current_block = holder.block
        elif statement.type == "expression_statement":
            # statement.children[0]: the expression
            # statement.children[1]: ;

            # Ignore empty statements. (i.e. just a semicolon)
            if len(statement.children) != 2:
                assert len(statement.children) == 1
                assert statement.children[0].type == ";"
            else:
                holder = BlockPointer(current_block)
                _ = bind_expression(statement.children[0], scope, holder, blocks)
                current_block = holder.block
        elif statement.type == "return_statement":
            if len(statement.children) <= 2:
                current_block.instructions.append(VarInstruction(RETURN_OP, None, [], statement))
            elif len(statement.children) == 3:
                assert statement.children[0].type == "return"
                assert statement.children[2].type == ";"
                holder = BlockPointer(current_block)
                return_value = expand_subexpression(statement.children[1], scope, holder, blocks)
                current_block = holder.block
                current_block.instructions.append(VarInstruction(RETURN_OP, None, [return_value], statement))
            else:
                raise AssertionError(f"Invalid number of fields for return_statement node: {len(statement.children)}")
            
            blocks.append((current_block, None)) # A block ending in a return statement has no successors.
            return blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "break_statement":
            current_block.instructions.append(VarInstruction(BREAK_OP, None, [], statement))
            blocks.append((current_block, BreakAssignment()))
            return blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "continue_statement":
            current_block.instructions.append(VarInstruction(CONTINUE_OP, None, [], statement))
            blocks.append((current_block, ContinueAssignment()))
            return blocks
        elif statement.type == "if_statement":
            # statement.children[0]: if
            # statement.children[1]: (condition) - the conditional test
            # statement.children[2]: (consequence) - the if statement body; entered if true.
            # may have
            # statement.children[3]: (alternative) - the body of the else branch.
            holder = BlockPointer(current_block)
            condition_result = expand_subexpression(get_child(statement, "condition"), scope, holder, blocks)
            current_block = holder.block
            current_block.instructions.append(VarInstruction(IF_OP, None, [condition_result], statement))

            # Computation that occurs inside the conditional statement can be added to the end of the current block.
            # Control flow ends the current basic block.
            # We add "None" here even though this block has successors because the code immediately below
            # ensures that its successors are assigned.
            blocks.append((current_block, None)) 
            if_start_block = current_block # Need to keep a reference to this around so we can assign the first body block to it.a
            current_block = BasicBlock([], [], []) # New basic block after control flow is complete.

            # Convert the if-statement body. Create a new scope for inside the if statement.
            body_blocks = convert_compound_statement(get_child(statement, "consequence"), Scope(scope))
            assert len(body_blocks) > 0, "Compound statement must have at least one corresponding basic block"
            if_start_block.add_successor(body_blocks[0][0])
            alternative_node = statement.child_by_field_name("alternative")
            if alternative_node is not None:
                # alternative_node.children[0]: else
                # alternative_node.children[1]: compound_statement or expression_statement: the else-clause body
                alternative_blocks = convert_compound_statement(alternative_node.children[1], Scope(scope))
                assert len(alternative_blocks) > 0, "Compound statement must have at least one corresponding basic block"
                if_start_block.add_successor(alternative_blocks[0][0])
                body_blocks.extend(alternative_blocks)
            else:
                if_start_block.add_successor(current_block)
            

            for ifblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    ifblock.add_successor(current_block)
                    blocks.append((ifblock, None))
                else:
                    blocks.append((ifblock, successor_assignment)) # propagate other successor assignments, like Break, outside this scope.
            
        elif statement.type == "for_statement":
            # The relevant children (i.e. non-syntax children) are
            # initializer, condition, update, body

            loop_scope = Scope(scope)

            initializer = statement.child_by_field_name("initializer")
            if initializer is not None:
                holder = BlockPointer(current_block)
                if initializer.type == "declaration":
                    loop_scope.parse_declaration(initializer, True, holder, blocks)
                else:
                    _ = bind_expression(initializer, loop_scope, holder, blocks)
                current_block = holder.block
            pre_loop_block = current_block # Keep a reference to the start of the loop arounds to assign successors later.
            blocks.append((pre_loop_block, None))
            current_block = BasicBlock([], [], []) # This is the block after the loop

            condition = statement.child_by_field_name("condition")
            condition_block = BasicBlock([], [], [])
            pre_loop_block.add_successor(condition_block) # must do this before expand_subexpression because it may change condition_block to the last block in the condition rather than the first.
            starting_condition_block = condition_block
            if condition is not None:
                holder = BlockPointer(condition_block)
                condition_result = expand_subexpression(condition, loop_scope, holder, blocks)
                condition_block = holder.block
                condition_block.instructions.append(VarInstruction(LOOP_OP, None, [condition_result], statement))
            else: # an empty loop condition
                condition_block.instructions.append(VarInstruction(LOOP_OP, None, [ONE], statement))
            blocks.append((condition_block, None))
            
            update = statement.child_by_field_name("update")
            if update is not None:
                update_block = BasicBlock([], [], [])
                starting_update_block = update_block # the update can itself contain control flow (via short-circuiting or a ternary) so we must keep a handle to the start of the update sub-CFG.
                update_holder = BlockPointer(update_block)
                _ = bind_expression(update, loop_scope, update_holder, blocks)
                update_block = update_holder.block
            else:
                starting_update_block = update_block = BasicBlock([], [], [])
            blocks.append((update_block, None))
            # at this point, update_block this is the ending update block.
            update_block.add_successor(starting_condition_block)

            body_blocks = convert_compound_statement(get_child(statement, "body"), loop_scope)
            condition_block.add_successor(body_blocks[0][0])
            condition_block.add_successor(current_block) # Do this after adding the body block to enforce the invariant that the true branch comes first.

            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(starting_update_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, ContinueAssignment):
                    loopblock.add_successor(starting_update_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, BreakAssignment):
                    loopblock.add_successor(current_block) # current_block is the block after the loop
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # propagate other successor assignments out of this block.
        elif statement.type == "while_statement":
            # statement.children[0]: while
            # statement.children[1]: condition
            # statement.children[2]: body
            pre_loop_block = current_block
            blocks.append((current_block, None))
            current_block = BasicBlock([], [], []) # The block after the while loop

            condition_block = BasicBlock([], [], [])
            starting_condition_block = condition_block # Always points to the first instead of the last condition block. With short-circuiting logical operators, the condition may be more than one block.
            pre_loop_block.add_successor(condition_block) # Important that we do this here because condition block is passed by reference to expand_subexpression and may be a different block after the call.
            condition_holder = BlockPointer(condition_block)
            condition_result = expand_subexpression(get_child(statement, "condition"), scope, condition_holder, blocks)
            condition_block = condition_holder.block
            condition_block.instructions.append(VarInstruction(LOOP_OP, None, [condition_result], statement))
            blocks.append((condition_block, None))

            loop_scope = Scope(scope)
            body_blocks = convert_compound_statement(get_child(statement, "body"), loop_scope)
            condition_block.add_successor(body_blocks[0][0]) # The first block in the compound statement is the first block of the loop.
            condition_block.add_successor(current_block) # Enforce the invariant that the true branch is first. current_block is the block after the while loop, which is entered on the false branch.

            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(starting_condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, ContinueAssignment):
                    loopblock.add_successor(starting_condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, BreakAssignment):
                    loopblock.add_successor(current_block) # break out of the loop; go to the block after the loop
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # propagate other successor assignments out of this block.
        elif statement.type == "do_statement":
            # statement.children[0]: do
            # statement.children[1]: (body) - the body of the loop
            # statement.children[2]: while
            # statement.children[3]: (condition) - the loop test
            # statement.children[4]: ;
            pre_loop_block = current_block # Keep a reference to the start of the loop arounds to assign successors later.
            blocks.append((pre_loop_block, None))
            current_block = BasicBlock([], [], []) # This is the block after the loop

            # Maintain a pointer to the starting condition block as we build the condition because the condition could contain multiple blocks (via short-circuiting logical operators or a ternary.)
            starting_condition_block = condition_block = BasicBlock([], [], [])
            condition_holder = BlockPointer(condition_block)
            condition_result = expand_subexpression(get_child(statement, "condition"), scope, condition_holder, blocks)
            condition_block = condition_holder.block
            condition_block.instructions.append(VarInstruction(LOOP_OP, None, [condition_result], statement))
            blocks.append((condition_block, None))

            loop_scope = Scope(scope)
            body_blocks = convert_compound_statement(get_child(statement, "body"), loop_scope)
            pre_loop_block.add_successor(body_blocks[0][0])
            condition_block.add_successor(body_blocks[0][0]) # Add this first to ensure that the true block is the first successor.
            condition_block.add_successor(current_block) # after the condition tests false, the loop exits to the next block after the loop.

            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(starting_condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, ContinueAssignment):
                    loopblock.add_successor(starting_condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, BreakAssignment):
                    loopblock.add_successor(current_block)
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # after the condition tests false, the loop exits to the next block after the loop.
        elif statement.type == "switch_statement":
            # condition_result is 'c' in switch (c) {...}
            holder = BlockPointer(current_block)
            condition_variable = expand_subexpression(get_child(statement, "condition"), scope, holder, blocks)
            current_block = holder.block

            blocks.append((current_block, None))
            current_if_block = current_block # Represents the block to which if statements will be added.
            prior_if_block: Optional[BasicBlock] = None # The if block representing the case condition before the current case condition.
            fallthrough_block: Optional[BasicBlock] = None # type: ignore # represents the previous block if there is no break statement at the end of that block.
            default_block: Optional[BasicBlock] = None # The first basic block from the default statement.

            # Contains the basic blocks from the individual case statements.
            switch_blocks: list[tuple[BasicBlock, BlockSuccessorAssignment | None]] = []

            # For variables declared inside of the switch statement
            switch_scope = Scope(scope)

            case_body = get_child(statement, "body").children
            assert case_body[0].type == "{" and case_body[-1].type == "}"
            for substatement in case_body[1:-1]:
                if substatement.type == "case_statement":
                    if substatement.children[0].type == "case":
                        comparison_value = check_expression_leaf(clean_expression(get_child(substatement, "value")), scope)
                        if not isinstance(comparison_value, IntegerConstant) and not isinstance(comparison_value, CharLiteral):
                            raise SemanticError("Case expression must be an integral constant expression.")
                        assert substatement.children[2].type == ":"

                        if prior_if_block is not None:
                            current_if_block = BasicBlock([], [], [])
                            switch_blocks.append((current_if_block, None))
                            prior_if_block.add_successor(current_if_block)

                        case_comparison_result = switch_scope.create_temporary()
                        current_if_block.instructions.append(VarInstruction(EqualTo(), case_comparison_result, [condition_variable, comparison_value], substatement)) # type: ignore
                        current_if_block.instructions.append(VarInstruction(IF_OP, None, [case_comparison_result], substatement)) # type: ignore

                        case_blocks = convert_compound_statement(substatement.children[3:], switch_scope)

                        # If the previous block is falling through, assign it as a successor.
                        if fallthrough_block is not None:
                            fallthrough_block.add_successor(case_blocks[0][0])
                        
                        current_if_block.add_successor(case_blocks[0][0]) # type: ignore

                        if isinstance(case_blocks[-1][1], Next):
                            # if this block doesn't end in a break or continue statement, fall through to the next case statement or default.
                            fallthrough_block: BasicBlock = case_blocks[-1][0]
                        else:
                            assert case_blocks[-1][1] is not None or case_blocks[-1][0].instructions[-1].op == RETURN_OP
                            # We needn't do anything here to handle break or continue statements; those will be handled as part of the 
                            # block successor assignment loop at the end of switch statement processing.
                            fallthrough_block = None # type: ignore

                        prior_if_block = current_if_block
                        current_if_block = None # Not strictly necessary; added for clarity.

                        switch_blocks.extend(case_blocks) # type: ignore
                    else:
                        assert substatement.children[0].type == "default" and substatement.children[1].type == ":"
                        assert default_block is None, "Cannot have more than one default statement in a switch statement."

                        default_blocks = convert_compound_statement(substatement.children[2:], switch_scope)
                        default_block = default_blocks[0][0]

                        # The previous case statement fell through into the default block.
                        if fallthrough_block is not None:
                            fallthrough_block.add_successor(default_block)

                        if isinstance(default_blocks[-1][1], Next):
                            fallthrough_block: BasicBlock = default_blocks[-1][0]
                        else:
                            assert default_blocks[-1][1] is not None or default_blocks[-1][0].instructions[-1].op == RETURN_OP
                            fallthrough_block = None # type: ignore
                        
                        switch_blocks.extend(default_blocks) # type: ignore
                elif substatement.type == "declaration":
                    switch_scope.parse_declaration(substatement, compile=False)
                # Anything other than a case/default statement or a declaration is ignored.
            
            # Represents the block after the switch statement. Only has to be a new block if the
            # switch statement is non-empty.
            if prior_if_block is not None or default_block is not None:
                current_block = BasicBlock([], [], [])

            # When all cases are exhausted, control should flow to the default block if it exists.
            # Otherwise, control should flow to the block after the switch statement.
            if default_block is not None:
                if prior_if_block is not None: # Can be None if there are no case statements in the switch.
                    prior_if_block.add_successor(default_block)
                else:
                    current_if_block.add_successor(default_block) # type: ignore
            elif prior_if_block is not None: # Can be None if there are no case statements in the switch.
                prior_if_block.add_successor(current_block)
            # else: The situation where there's no default or case statements requires no successor assignment.

            # A break statement will have a link from the last case to the post-switch block
            # as well, but that's handled by the block successor assignment loop below.
            if fallthrough_block is not None:
                fallthrough_block.add_successor(current_block)

            for switch_block, successor_assignment in switch_blocks:
                if isinstance(successor_assignment, BreakAssignment):
                    assert len(switch_block.instructions) > 0 and isinstance(switch_block.instructions[-1].op, Break)
                    switch_block.instructions.pop() # remove the break statement because it has loop-associated semantics in the interpreter, whereas here it's just a successor assignment.
                    switch_block.add_successor(current_block)
                    blocks.append((switch_block, None))
                elif isinstance(successor_assignment, Next):
                    # "Next" successor assignment are dealt with above through fallthrough_block links.
                    blocks.append((switch_block, None))
                else:
                    blocks.append((switch_block, successor_assignment))
        elif statement.type == "compound_statement":
            # This is just a plain compound statement, not associated with a loop 
            # if statement, or other type of statement.
            
            blocks.append((current_block, None))
            # Model the separate scope introduced by a compound statement with a new Scope object.
            nested_blocks = convert_compound_statement(statement, Scope(scope))
            current_block.add_successor(nested_blocks[0][0])
            current_block = BasicBlock([], [], []) # The block after the inner compound statement ends.
            for nested_block, successor_assignment in nested_blocks:
                if isinstance(successor_assignment, Next):
                    nested_block.add_successor(current_block)
                    blocks.append((nested_block, None))
                else:
                    blocks.append((nested_block, successor_assignment))
            if nested_blocks[-1][1] is None:
                assert nested_blocks[-1][0].instructions[-1].op == RETURN_OP
                # A return statement in a nested scope terminates the outer scope as well.
                return blocks # propagate the return's termination of this path to this scope.
        elif statement.type == "comment":
            pass
        elif statement.type == "struct_specifier":
            pass
        elif statement.type == ";":
            pass
        elif statement.type == "ERROR":
            raise ParsingError(get_text(statement))
        # TODO: Implement processing for other types of statements.
        else:
            raise NotImplementedError(f"No code for handling statements of type {statement.type}")
    
    blocks.append((current_block, Next()))

    return blocks

def error_check(node: Node):
    """Determine if there is an error node in this AST. If there is, raise a ParsingError.
    """
    if node.type == "ERROR":
        raise ParsingError(get_text(node))
    
    for child in node.children:
        error_check(child)

def clean_up_empty_blocks(blocks: list[BasicBlock]):
    remove_blocks: set[BasicBlock] = set()
    # Ignore the entry block; it can be validly empty if the first statement in the function is a loop.
    for block in itertools.islice(blocks, 1, None):
        if len(block.instructions) == 0:
            # Having more would require a branch, which would mean that the basic block is not empty.
            assert len(block.successors) <= 1, "An empty basic block should have one or fewer successors."
            successor = block.successors[0] if len(block.successors) == 1 else None

            # Only remove empty blocks if doing so would not decrease the number of successors of any predecessor block that
            # ends with a branch (i.e. with two or more successors.) Doing so affects the control flow graph, which may lead
            # to incorrect control dependence relationships.
            if any(len(predecessor.successors) > 1 and (successor is None or successor in predecessor.successors) 
                   for predecessor in block.predecessors):
                continue

            for predecessor in block.predecessors:
                # Find this block in the predecessor's successor list.
                this_block_index = None
                for i, pred_suc in enumerate(predecessor.successors):
                    if pred_suc == block:
                        assert this_block_index is None, "A basic block cannot have the same successor multiple times!"
                        this_block_index = i
                assert this_block_index is not None # should always be true, but for mypy
                
                if successor is None or successor in predecessor.successors:
                    # sucessor in predecessor.successors can happen if block's predecessor also has block's successor
                    # as a successor. This happens in an if statement with an empty body.
                    del predecessor.successors[this_block_index]
                else:
                    predecessor.successors[this_block_index] = successor

            if successor is not None:
                # Also ensure that the empty block's successor's predecessor list is updated.
                successor.predecessors.remove(block) # Remove the empty block...
                # ...and replace it with all of the predecessors of the empty block
                for empty_predecessor in block.predecessors:
                    if empty_predecessor not in successor.predecessors:
                        successor.predecessors.append(empty_predecessor)
            
            # Remove these later so as not to damage the iterator.
            assert block not in remove_blocks, "A basic block should not be listed multiple times in a function's list of basic blocks!"
            remove_blocks.add(block)
    
    for block in remove_blocks:
        blocks.remove(block)

def remove_unreachable_blocks(fn: Function):
    """Remove basic blocks that are unreachable from the entry block. The modification
    is performed in-place.

    :param function: The function for which unreachable blocks should be removed.
    """
    reachable: set[BasicBlock] = set()

    def search(bb: BasicBlock):
        if bb in reachable:
            return
        reachable.add(bb)
        for successor in bb.successors:
            search(successor)
    search(fn.entry_block)

    # Only rebuild the basic_block list if we have to
    if len(reachable) < len(fn.basic_blocks):
        new_blocks = []
        for block in fn.basic_blocks:
            if block in reachable:
                new_blocks.append(block)
            else:
                # unreachable blocks may have successors that are reachable.
                for successor in block.successors:
                    successor.predecessors.remove(block)
        fn.basic_blocks = new_blocks

def convert_function(definition: Node, global_scope: Scope):
    """Converts a tree-sitter AST for a function definition into codealign IR variable form.

    :param definition: The root node of the function definition. Should be of type 'function_definition'.
    """
    assert(definition.type == "function_definition")

    # Tree-sitter can effectively recover from some errors, but other times it inserts an ERROR node.
    # This can cause problems for generating IR. Thus, we do an initial check to make sure that the
    # given AST does not have any ERROR nodes.
    error_check(definition)

    function_scope = Scope(global_scope) # The highest level scope in the function itself.

    # We parse the signature on the global scope because a defined function is a symbol that is globally available.
    # Note that parse_function_signature will add any types defined within the signature to the scope, which means
    # that if any types are defined within the function signature, they'll be added to the global scope instead of
    # the function scope, as is proper. However, given that this is useless (one cannot actually _call_ such a function
    # because one cannot create a variable of the correct type because the type definition can only be accessed within
    # the function), we won't worry too much about it here.
    typ, function_name = global_scope.parse_function_signature(definition)

    # We add the parameters to the function scope because the parameters are only accessible within the function
    parameters = []
    for t, n in typ.parameters:
        if not isinstance(t, FunctionType.VariadicParameter):
            # This shouldn't be possible in a function definition, but we'll assert it to be sure and make mypy happy.
            assert n is not None, f"Missing a name in parameter list for function definition:\n\n{get_text(definition)}"
            parameters.append(function_scope.add_parameter(t, n))

    # definition.children[2] (body) is the function body. We need to convert all statements in the body into instructions.
    blocks_with_metadata = convert_compound_statement(get_child(definition, "body"), function_scope)

    basic_blocks = [b[0] for b in blocks_with_metadata]
    clean_up_empty_blocks(basic_blocks)

    func = Function(function_name, basic_blocks, parameters, typ.return_type, definition)
    remove_unreachable_blocks(func)
    return func

    
def compile(code: bytes, short_circuit_logical_ops: bool = True) -> list[Function[VarInstruction]]:
    """Parse C code using tree-sitter, and convert it into variable-oriented IR function form.
    """
    root = parser.parse(code).root_node
    assert root.type == "translation_unit"

    global_scope = Scope(short_circuit_logical_ops=short_circuit_logical_ops)
    functions = []
    for child in root.children:
        if child.type == "function_definition":
            functions.append(convert_function(child, global_scope))
        else:
            global_scope.record_item(child)
    
    return functions
