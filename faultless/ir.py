"""Intermediate representations of C code.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Iterable, TypeVar, Optional, Literal, Generic, Sequence, TypeGuard, overload
from typing import Union as tUnion
from collections import deque, Counter
import itertools
import contextlib
import warnings

from tree_sitter import Node
import z3

### Exceptions
class FaultlessError(Exception):
    """Base class for all custom faultless exceptions."""

class SemanticError(FaultlessError):
    """Represents a violation of C semantics."""

class TypeDeductionError(SemanticError):
    """Thrown when a type issue is detected in type deduction."""

class ExecutionError(FaultlessError):
    """An error reflecting a fundamental limit with the execution model with the information given."""

class UnsupportedFeatureError(FaultlessError):
    """The specified code feature is not supported."""

class UnknownSMTResultError(FaultlessError):
    """The satisfiability of a formula is z3.unknown"""

MAX_ARRAY_READ_SIZE: int | None = None

#####################
# Type object model #
#####################
# The type object model is deliberately more permissive than C's actualy type system.
# In particular, incomplete types are explicitly allowed in derived types, which is only
# allowable in standard C for pointers. This restriction exists because variables need to 
# have a defined size to be compiled/laid out in memory. (Exceptions like variable-length
# arrays exist.) However, codealign is designed to work on code fragments in isolation,
# without access to the rest of the project, which means type definitions might not 
# always be available. Thus, some flexibility is required.

class CType:
    def __hash__(self):
        raise NotImplementedError("Unhashable object: Abstract CType")

    def __eq__(self, other: Any):
        raise NotImplementedError("__eq__: Cannot compare abstract CType object.")
    
    def __str__(self):
        return self.declaration("")
    
    def __repr__(self):
        return str(self)
    
    def declaration(self, decl: str) -> str:
        raise NotImplementedError("Abstract CType object does not have a declaration.")

    def stubify(self) -> "CType":
        """Return the minimum form of the type assuming all other relevant types are defined elsewhere.
        In other words, a stubified type contains the most incomplete types possible.
        """
        return self

class IncompleteType(CType):
    """Contains information about an incomplete type. These don't have a definition or, implicitly, a size.
    """
    
class IncompleteUDT(IncompleteType):
    def __init__(self, name: str):
        self.name = name

class IncompleteStruct(IncompleteUDT):
    def __init__(self, name: str, full_definition: "Struct | None" = None):
        super().__init__(name)
        self.full_definition = full_definition

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, IncompleteStruct) and other.name == self.name

    def __str__(self):
        return f"struct {self.name}"
    
    def declaration(self, decl: str) -> str:
        if decl == "":
            return f"struct {self.name}"
        else:
            return f"struct {self.name} {decl}"
        
    def typeof(self, member_name: str) -> CType | None:
        assert self.full_definition is not None
        self.full_definition.typeof(member_name)

    def offsetof(self, field: "Field | str") -> int | None:
        assert self.full_definition is not None
        self.full_definition.offsetof(field)

    def get_size(self) -> int:
        assert self.full_definition is not None
        return self.full_definition.get_size()
    
class IncompleteEnum(IncompleteUDT):
    def __init__(self, name: str, full_definition: "Enum | None" = None):
        super().__init__(name)
        self.full_definition = full_definition

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, IncompleteEnum) and other.name == self.name

    def __str__(self):
        return f"enum {self.name}"
    
    def declaration(self, decl: str) -> str:
        if decl == "":
            return f"enum {self.name}"
        else:
            return f"enum {self.name} {decl}"
    
class IncompleteUnion(IncompleteUDT):
    def __init__(self, name: str, full_definition: "Union | None" = None):
        super().__init__(name)
        self.full_definition = full_definition

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, IncompleteUnion) and other.name == self.name

    def __str__(self):
        return f"union {self.name}"
    
    def declaration(self, decl: str) -> str:
        if decl == "":
            return f"union {self.name}"
        else:
            return f"union {self.name} {decl}"
        
    def typeof(self, member_name: str) -> CType | None:
        assert self.full_definition is not None
        return self.full_definition.typeof(member_name)
    
    def get_size(self):
        assert self.full_definition is not None
        return self.full_definition.get_size()
        
class Void(IncompleteType):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Void)

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return "void"
    
    def declaration(self, decl: str) -> str:
        if decl == "":
            return "void"
        return f"void {decl}"
    
    def stubify(self) -> "Void":
        return self
    
class UnknownType(CType):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UnknownType)
    
    def __hash__(self) -> int:
        return 1
    
    def __str__(self) -> str:
        return "unknown"
    
    def declaration(self, decl: str) -> str:
        if decl == "":
            return "unknown"
        return f"unknown {decl}"
    
    def stubify(self) -> "UnknownType":
        return self
    
class FunctionType(CType):
    class VariadicParameter:
        def __init__(self):
            pass
        
        def __str__(self) -> str:
            return "..."
        
        def __eq__(self, other) -> bool:
            return isinstance(other, FunctionType.VariadicParameter)

        def __hash__(self) -> int:
            return 15
        
        def declaration(self, decl: str):
            assert len(decl) == 0, f"Variadic parameters cannot have a name!"
            return str(self)
        
        def stubify(self) -> "FunctionType.VariadicParameter":
            return self

    void_decl = [(Void(), None)]

    def __init__(self, return_type: CType, parameters: list[tuple["CType | VariadicParameter", str | None]]):
        self.return_type = return_type
        if parameters == FunctionType.void_decl:
            self.parameters = []
        else:
            self.parameters = parameters
    
    def __hash__(self) -> int:
        value = hash(self.return_type)
        for typ, name in self.parameters:
            value ^= hash(typ)
            if name is not None:
                value ^= hash(name)
        return value + 14

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FunctionType):
            return self.return_type == other.return_type and self.parameters == other.parameters
        return False

    def __str__(self) -> str:
        def var_desc(tp: "CType | FunctionType.VariadicParameter", var: str | None) -> str:
            if var is not None:
                return f"{tp} {var}"
            elif isinstance(tp, FunctionType.VariadicParameter):
                return "..."
            else:
                return str(tp)
        return f"{self.return_type} (" + ", ".join(var_desc(*param) for param in self.parameters) + ")"

    # def __repr__(self) -> str:
    #     return f"FunctionType(return_type={repr(self.return_type)}, parameters=(" + ", ".join(repr(tp) + " " + str(var) for tp, var in self.parameters) + "))" 
    
    def _decl_str(self, decl: str) -> str:
        parameters: list[str] = []
        for p_type, p_name in self.parameters:
            if p_name is None:
                p_name = ""
            parameters.append(p_type.declaration(p_name))
        return f"{decl}(" + ", ".join(parameters) + ")"
    
    def declaration(self, decl: str) -> str:
        return self.return_type.declaration(self._decl_str(decl))
    
    def stubify(self) -> "FunctionType":
        return FunctionType(
            self.return_type.stubify(),
            [(typ.stubify(), n) for typ, n in self.parameters]
        )
    
class ObjectType(CType):
    """A type which has a defined size according to the C standard. (Codealign's type system
    relaxes this constraint.)
    """
    def __init__(self):
        raise NotImplementedError(f"Cannot instantiate an abstract ObjectType")
    
    def get_size(self) -> int:
        raise NotImplementedError(f"Cannot get the size for an abstract ObjectType")

class PrimitiveType(ObjectType):
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size # important for primitive types

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PrimitiveType):
            return type(self) == type(other) and self.name == other.name and self.size == other.size
        return False
    
    def __hash__(self) -> int:
        return hash((self.name, self.size))

    def __str__(self) -> str:
        return f"{self.name}"

    def declaration(self, decl: str) -> str:
        if decl == "":
            return self.name
        return f"{self.name} {decl}"
    
    def get_size(self) -> int:
        return self.size
    
class Float(PrimitiveType):
    pass

class Integer(PrimitiveType):
    pass

class SignedInteger(Integer):
    pass

class UnsignedInteger(Integer):
    pass

######### Important Integer Sizes ##########
INTEGER = SignedInteger("int", 4) # There are numerous types that default to 'int'.
SIZE_T = UnsignedInteger("unsigned long", 8) # For pointer sizes
CHARACTER = SignedInteger("char", 1)

LARGE_ARRAY_READ_WARN_THRESHOLD = 32
    
class Array(ObjectType):
    """Represents an array.
    
    To help codealign support underspecified code fragments, arrays support arbitrary CTypes, including IncompleteTypes.
    """
    def __init__(self, element_type: CType, nelements: int):
        """
        :param nelements: the number of elements in the array
        :param element_type: the type of each element. A string is accepted to keep this class backwards-compatible with DIRTY-generated data.
        """
        self.element_type = element_type
        self.nelements = nelements
        if nelements < 0:
            raise SemanticError(f"Array has a negative number of elements: {nelements}")

    def __eq__(self, other) -> bool:
        if isinstance(other, Array):
            return (
                self.nelements == other.nelements
                and self.element_type == other.element_type # This should imply that the sizes are equal. This also lets the __eq__ method work for incomplete types
            )
        return False

    def __hash__(self) -> int:
        return hash((self.nelements, self.element_type))
    
    def _decl_str(self, decl: str) -> str:
        if self.nelements == 0:
            return f"{decl}[]"
        return f"{decl}[{self.nelements}]"

    def declaration(self, decl: str) -> str:
        return self.element_type.declaration(self._decl_str(decl))
    
    def stubify(self) -> "Array":
        element_type = self.element_type.stubify()
        return Array(nelements=self.nelements, element_type=element_type)
    
    def get_size(self) -> int:
        return self.get_element_size() * self.nelements
    
    def get_element_size(self):
        if not isinstance(self.element_type, ObjectType):
            raise ExecutionError(f"Cannot get the size of an array with element type {self.element_type}")
        return self.element_type.get_size()
    
class Pointer(ObjectType):
    """Stores information about a pointer.
    """

    size = SIZE_T.size

    def __init__(self, target_type: CType):
        self.target_type = target_type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Pointer):
            return self.target_type == other.target_type
        return False

    def __hash__(self) -> int:
        # "1 +" to make it different from the hash of the target type.
        return 1 + hash(self.target_type)
    
    # def __repr__(self) -> str:
    #     return f"Pointer({repr(self.target_type)})"
    
    def _decl_str(self, decl: str) -> str:
        if isinstance(self.target_type, Array) or isinstance(self.target_type, FunctionType):
            return f"(*{decl})"
        return f"*{decl}"
    
    def declaration(self, decl: str) -> str:
        return self.target_type.declaration(self._decl_str(decl))
    
    def stubify(self) -> "Pointer":
        return Pointer(self.target_type.stubify())
    
    def get_size(self) -> int:
        return SIZE_T.size
    
class UDT:
    """An object representing a struct, union, or enum"""

    def __init__(self, name: str | None):
        self.name = name

    class Field:
        """Information about a field in a struct or union"""
        def __init__(self, type: CType, name: str):
            self.name = name
            self.type = type

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, UDT.Field):
                return self.name == other.name and self.type == other.type
            return False

        def __hash__(self) -> int:
            return hash((self.name, self.type))

        def __str__(self) -> str:
            return self.type.declaration(self.name)
        
        def declaration(self) -> str:
            return self.type.declaration(self.name)
    
    @property
    def stub(self) -> IncompleteType:
        raise NotImplementedError("Cannot create stub for an abstract UDT.")
    
class Struct(ObjectType, UDT):
    """Stores information about a struct"""
    ALIGNMENT_SIZE = 8

    def __init__(self, name: str | None, members: "Iterable[UDT.Field]", defer_layout: bool = False):
        UDT.__init__(self, name)
        self.members = tuple(members)
        try:
            self.fieldname2offset, self.offset2field, self.size = self._compute_layout()
        except ExecutionError:
            if not defer_layout:
                raise
            self.fieldname2offset = {}
            self.offset2field = {}
            self.size = -1

    def _compute_layout(self) -> tuple[dict[str, int], dict[int, UDT.Field], int]:
        offsets: dict[str, int] = {}
        fields: dict[int, UDT.Field] = {}
        offset = 0
        max_alignment = 1

        for member in self.members:
            if not isinstance(member.type, ObjectType):
                raise ExecutionError(f"Cannot compute layout of {self} due to non-object member {member}")

            member_size = member.type.get_size()
            alignment_size = member.type.get_element_size() if isinstance(member.type, Array) and member.type.nelements == 0 else member_size
            alignment = min(alignment_size, Struct.ALIGNMENT_SIZE)
            assert alignment > 0, f"Invalid alignment for struct member {member} in {self}."
            max_alignment = max(max_alignment, alignment)

            if offset % alignment != 0:
                offset += alignment - (offset % alignment)

            offsets[member.name] = offset
            fields[offset] = member
            offset += member_size

        if offset % max_alignment != 0:
            offset += max_alignment - (offset % max_alignment)

        return offsets, fields, offset

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Struct):
            return self.name == other.name and self.members == other.members
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.members))

    def __str__(self) -> str:
        if self.name is None:
            ret = f"struct {{ "
        else:
            ret = f"struct {self.name} {{ "
        for l in self.members:
            ret += f"{str(l)}; "
        ret += "}"
        return ret
    
    def declaration(self, decl: str) -> str:
        start = "struct { " if self.name is None else f"struct {self.name} {{ "
        specifier = start + "; ".join(
            (l.declaration() if isinstance(l, UDT.Field) else l.declaration(""))
            for l in self.members
        ) + (";" if len(self.members) > 0 else "") + " }"

        if decl == "":
            return specifier
        else:
            return f"{specifier} {decl}"
        
    def stubify(self) -> "IncompleteStruct | Struct":
        # The most incomplete form of an anonymous struct is the full definition, with its fields incomplete as possible.
        if self.name is None:
            layout = []
            for l in self.members:
                if isinstance(l, (Struct, Union)):
                    layout.append(l.stubify())
                elif isinstance(l, UDT.Field):
                    layout.append(UDT.Field(name=l.name, type=l.type.stubify()))
            return Struct(name=self.name, members=layout)
        return self.stub
    
    @property
    def stub(self) -> IncompleteStruct:
        if self.name is None:
            raise ValueError("Cannot create incomplete type for an anonymous struct.")
        return IncompleteStruct(self.name)
    
    def typeof(self, member_name: str) -> CType | None:
        offset = self.fieldname2offset.get(member_name)
        if offset is not None:
            return self.offset2field[offset].type
        return None
    
    def offsetof(self, field: "Field | str") -> int | None:
        """Get the offset of the field with the provided field name, or return None if it is not found."""
        return self.fieldname2offset.get(field.value if isinstance(field, Field) else field)
    
    def get_size(self) -> int:
        if self.size < 0:
            raise ExecutionError(f"Cannot get the size for incompletely expanded struct {self}")
        return self.size
    
class Union(ObjectType, UDT):
    def __init__(self, name: str | None, members: "Iterable[UDT.Field]"):
        UDT.__init__(self, name)
        self.members = tuple(members)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Union):
            return (
                self.name == other.name
                and self.members == other.members
            )
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.members))

    def __str__(self) -> str:
        if self.name is None:
            ret = f"union {{ "
        else:
            ret = f"union {self.name} {{ "
        for m in self.members:
            ret += f"{str(m)}; "
        ret += "}"
        return ret
    
    def declaration(self, decl: str) -> str:
        start = "union { " if self.name is None else f"union {self.name} {{ " 
        specifier = start + "; ".join(
            m.declaration() if isinstance(m, UDT.Field) else m.declaration("")
            for m in self.members
        ) + "; }"
        if decl == "":
            return specifier
        else:
            return specifier + " " + decl
        
    def stubify(self) -> "IncompleteUnion | Union":
        if self.name is None:
            members = []
            for m in self.members:
                if isinstance(m, (Struct, Union)):
                    members.append(m.stubify())
                else:
                    members.append(UDT.Field(name=m.name, type=m.type.stubify()))
            return Union(name=self.name, members=members)
        return self.stub
    
    @property
    def stub(self) -> IncompleteUnion:
        if self.name is None:
            raise ValueError("Cannot create incomplete type for an anonymous union")
        return IncompleteUnion(self.name)
    
    def typeof(self, member_name: str) -> CType | None:
        for member in self.members:
            if member.name == member_name:
                return member.type
        return None
    
    def get_size(self) -> int:
        sizes = []
        for m in self.members:
            if isinstance(m.type, ObjectType):
                sizes.append(m.type.get_size())
            else:
                raise ExecutionError(f"Cannot get size for {self} due to non-object member {m}")
        return max(sizes)


class Enum(ObjectType, UDT):
    size = 4 # enums are just syntactic sugar for ints

    class Member:
        def __init__(self, *, name: str, value: int | None):
            self.name = name
            self.value = value # For now, can be None when there is an expression initializing the member.

        def __hash__(self) -> int:
            return hash(self.name) + (0 if self.value is None else self.value)
        
        def __eq__(self, other) -> bool:
            return isinstance(other, Enum.Member) and self.name == other.name and self.value == other.value

        def __str__(self):
            if self.value is None:
                return f"{self.name}=<expr>"
            return f"{self.name}={self.value}"

    def __init__(self, *, name: str | None, members: list[Member]):
        self.members = members
        self.name = name

    def __hash__(self) -> int:
        val = hash(self.name)
        for m in self.members:
            val ^= hash(m)
        return val

    def __eq__(self, other):
        return isinstance(other, Enum) and self.name == other.name and all(m1 == m2 for m1, m2 in zip(self.members, other.members))

    def __str__(self):
        return f"enum {self.name} {{" + ", ".join(str(m) for m in self.members) + "}"

    def declaration(self, decl: str) -> str:
        start = "enum { " if self.name is None else f"enum {self.name} {{ "
        specifier = start + ", ".join(
            m.name + "=" + ("expr" if m.value is None else str(m.value))
            for m in self.members
        ) + " }"
        if decl == "":
            return specifier
        else:
            return specifier + " " + decl
        
    def stubify(self) -> "IncompleteEnum | Enum":
        if self.name is None:
            return self
        return self.stub
    
    @property
    def stub(self) -> IncompleteEnum:
        if self.name is None:
            raise ValueError("Cannot create incomplete type for an anonymous enum")
        return IncompleteEnum(self.name)
    
    def get_size(self) -> int:
        return INTEGER.size
    
#########################
# End Type object model #
#########################


#
# Constants
#
class Constant(ABC):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

class IntegerConstant(Constant):
    def __init__(self, value: int, type: Integer):
        self.value = value
        self.type = type

# Important integer constants referenced by the language definition.
ONE = IntegerConstant(1, INTEGER)
ZERO = IntegerConstant(0, INTEGER)

class FloatConstant(Constant):
    def __init__(self, value: float, type: Float):
        self.value = value
        self.type = type

class CharLiteral(Constant):
    def __init__(self, value: int, type: SignedInteger = INTEGER): # a character literal as used in code is defined as an int.
        self.value = value
        self.type = type

    def __eq__(self, other) -> bool:
        return isinstance(other, CharLiteral) and self.value == other.value and self.type == other.type

    def __hash__(self) -> int:
        return hash(self.value) ^ hash(self.type)

    def __repr__(self):
        if self.value == 0:
            return "'\0'" # This is a very common case and the chr representation is pretty bad ('\x00')
        return f"'{chr(self.value)}'"

class StringLiteral(Constant):
    def __init__(self, value: str, characters: Iterable[CharLiteral]):
        self.value = value
        self.characters = tuple(characters)
        self.type = Array(CHARACTER, len(self.characters))

    def __len__(self):
        return len(self.characters)

    def __eq__(self, other):
        return isinstance(other, StringLiteral) and self.characters == other.characters
    
    def __hash__(self) -> int:
        return hash(self.characters)

    def __repr__(self):
        return f"\"{self.value}\""
    
    def __str__(self):
        return repr(self)

# Struct field
class Field(Constant):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return self.value

# For situations where variables need to be tracked but don't have values
class Uninitialized(Constant):
    def __init__(self, variable: "Variable"):
        self.value = variable

    def __str__(self):
        return f"<Uninitialized({self.value})>"

#
# Variables
#
CTypeT = TypeVar("CTypeT", bound=CType)
class Variable(Generic[CTypeT]):
    def __init__(self, type: CTypeT, name: str, is_temporary: bool=False, is_stack_allocated: bool=True):
        self.name = name
        self.type = type
        self.is_temporary = is_temporary
        self.is_stack_allocated = is_stack_allocated
    
    def __eq__(self, other: 'Variable'):
        return id(self) == id(other)

    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return id(self)

class Parameter(Variable):
    def __init__(self, type: CType, name: str):
        super().__init__(type, name, is_temporary=False)

class GlobalVariable(Variable):
    def __init__(self, type: CType, name: str):
        super().__init__(type, name, is_temporary=False)


DECOMPILER_PLACEHOLDER_TYPES: set[Integer] = set()
def set_decompiler_placeholder_types(types: Iterable[Integer] | Literal['Hex-Rays']):
    """Set types that a decompiler uses to indicate a type of an integral type of a given size without to committing to a specific one.

    There are pre-sets for specific decompilers, or supply your own as an iterable of integers.
    """
    if isinstance(types, str) and types == "Hex-Rays":
        types = [
            SignedInteger("BYTE", 1), SignedInteger("WORD", 2), SignedInteger("DWORD", 4), SignedInteger("QWORD", 8),
            UnsignedInteger("_BYTE", 1), UnsignedInteger("_WORD", 2), UnsignedInteger("_DWORD", 4), UnsignedInteger("_QWORD", 8)
        ]
    else:
        for t in types:
            assert isinstance(t, Integer), f"Decompiler placeholder types must be instances of faultless.ir.Integer but found {t} ({type(t)})"
    DECOMPILER_PLACEHOLDER_TYPES.update(types)

def is_decompiler_placeholder_type(t: CType) -> TypeGuard[Integer]:
    return t in DECOMPILER_PLACEHOLDER_TYPES


################
# Memory Model #
################

#
# Z3 utilities
#
SymbolicExpression = z3.ArithRef | z3.BitVecRef | z3.FPRef

Z3ReprKind = Literal["bitvec", "int"]
Z3FloatReprKind = Literal["real", "float"]

class Z3ReprOptions:
    """Controls how C scalars and pointers are represented in Z3."""
    def __init__(self, integer_repr: Z3ReprKind = "bitvec", pointer_repr: Z3ReprKind = "int", float_repr: Z3FloatReprKind = "real"):
        self.integer_repr = integer_repr
        self.pointer_repr = pointer_repr
        self.float_repr = float_repr
        self._validate()

    def _validate(self) -> None:
        if self.integer_repr not in ("bitvec", "int"):
            raise ValueError(f"Invalid integer_repr: {self.integer_repr}")
        if self.pointer_repr not in ("bitvec", "int"):
            raise ValueError(f"Invalid pointer_repr: {self.pointer_repr}")
        if self.float_repr not in ("real", "float"):
            raise ValueError(f"Invalid float_repr: {self.float_repr}")

Z3_REPR_OPTIONS = Z3ReprOptions()

def set_z3repr_options(*, integer_repr: Z3ReprKind | None = None, pointer_repr: Z3ReprKind | None = None, float_repr: Z3FloatReprKind | None = None) -> None:
    """Update how integers, pointers, and floats are represented in Z3."""
    if integer_repr is not None:
        Z3_REPR_OPTIONS.integer_repr = integer_repr
    if pointer_repr is not None:
        Z3_REPR_OPTIONS.pointer_repr = pointer_repr
    if float_repr is not None:
        Z3_REPR_OPTIONS.float_repr = float_repr
    Z3_REPR_OPTIONS._validate()
    # Keep derived constants in sync with the active representation.
    global Z3_ONE, Z3_ZERO
    Z3_ONE = z3repr(ONE)
    Z3_ZERO = z3repr(ZERO)

@contextlib.contextmanager
def z3repr_options(*, integer_repr: Z3ReprKind | None = None, pointer_repr: Z3ReprKind | None = None, float_repr: Z3FloatReprKind | None = None):
    """Temporarily override Z3 representation settings within a context."""
    prior_integer: Z3ReprKind = Z3_REPR_OPTIONS.integer_repr # type: ignore
    prior_pointer: Z3ReprKind = Z3_REPR_OPTIONS.pointer_repr # type: ignore
    prior_float: Z3FloatReprKind = Z3_REPR_OPTIONS.float_repr # type: ignore
    set_z3repr_options(integer_repr=integer_repr, pointer_repr=pointer_repr, float_repr=float_repr)
    try:
        yield
    finally:
        set_z3repr_options(integer_repr=prior_integer, pointer_repr=prior_pointer, float_repr=prior_float)

def _pointer_bitwidth() -> int:
    return SIZE_T.size * 8

def _z3_float_sort(size_bytes: int) -> z3.FPSortRef:
    match size_bytes:
        case 4:
            return z3.Float32()
        case 8:
            return z3.Float64()
        case 16:
            return z3.FPSort(15, 113)
        case _:
            raise NotImplementedError(f"Unsupported float size: {size_bytes} bytes")

def _z3_integer_value(value: int, size_bytes: int) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.integer_repr == "bitvec":
        return z3.BitVecVal(value, size_bytes * 8)
    return z3.IntVal(value)

def _z3_integer_symbol(name: str, size_bytes: int) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.integer_repr == "bitvec":
        return z3.BitVec(name, size_bytes * 8)
    return z3.Int(name)

def _z3_pointer_value(value: int) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.pointer_repr == "bitvec":
        return z3.BitVecVal(value, _pointer_bitwidth())
    return z3.IntVal(value)

def _z3_pointer_symbol(name: str) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.pointer_repr == "bitvec":
        return z3.BitVec(name, _pointer_bitwidth())
    return z3.Int(name)

def _z3_float_value(value: float, size_bytes: int) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.float_repr == "float":
        return z3.FPVal(value, _z3_float_sort(size_bytes))
    return z3.RealVal(str(value))

def _z3_float_symbol(name: str, size_bytes: int) -> SymbolicExpression:
    if Z3_REPR_OPTIONS.float_repr == "float":
        return z3.FP(name, _z3_float_sort(size_bytes))
    return z3.Real(name)

def _truncate_real_to_int(expr: z3.ArithRef) -> z3.ArithRef:
    return z3.If(expr >= 0, z3.ToInt(expr), -z3.ToInt(-expr)) # type: ignore

def z3_zero(t: CType) -> "Value":
    """Return type t, but set to zero."""
    if isinstance(t, Integer):
        return Value.make(IntegerConstant(0, t))
    elif isinstance(t, Float):
        return Value.make(FloatConstant(0, t))
    elif isinstance(t, Pointer):
        return Value(t, _z3_pointer_value(0))
    elif isinstance(t, Array):
        zero = z3_zero(t.element_type)
        element_size = t.get_element_size()
        return CompoundValue(t, {i * element_size: zero for i in range(t.nelements)})
    elif isinstance(t, Struct):
        return CompoundValue(t, {offset: z3_zero(field.type) for offset, field in t.offset2field.items()})
    elif isinstance(t, Union):
        return z3_zero(t.members[0].type)
    raise ExecutionError(f"Cannot construct a zero element for type {t}")

def truthiness(value: "Value") -> z3.BoolRef | bool:
    if isinstance(value, ConditionalValue):
        return value.condition
    if isinstance(value.type, (Integer, Pointer)):
        return value.expr != 0
    if isinstance(value.type, Float):
        if Z3_REPR_OPTIONS.float_repr == "float":
            return value.expr != z3.FPVal(0.0, _z3_float_sort(value.type.size))
        return value.expr != z3.RealVal("0")
    raise SemanticError(f"Cannot determine truthiness of non-scalar type {value.type}")

def _bitvec_resize(expr: z3.BitVecRef, from_bits: int, to_bits: int, signed: bool) -> z3.BitVecRef:
    if from_bits == to_bits:
        return expr # the bit pattern is the same, it's just interpreted differently.   
    elif from_bits < to_bits:
        extend = to_bits - from_bits
        return z3.SignExt(extend, expr) if signed else z3.ZeroExt(extend, expr)
    else:
        return z3.Extract(to_bits - 1, 0, expr) # type: ignore

def is_z3_numeric_literal(expr: z3.ExprRef) -> bool:
    return z3.is_int_value(expr) or z3.is_bv_value(expr) or z3.is_rational_value(expr) or z3.is_fp_value(expr) or z3.is_algebraic_value(expr)

def is_z3_variable(expr: z3.ExprRef) -> bool:
    return z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED

def z3_contains_variable(variable: SymbolicExpression, expr: SymbolicExpression) -> bool:
    assert is_z3_variable(variable), f"Expected a Z3 variable but found {variable}."
    return any(variable.eq(expr_var) for expr_var in z3.z3util.get_vars(expr))

@overload
def substitute_z3_expr(expr: z3.BoolRef | bool, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> z3.BoolRef | bool: ...
@overload
def substitute_z3_expr(expr: SymbolicExpression, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> SymbolicExpression: ...
def substitute_z3_expr(expr, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]):
    """Replace each instance of the symbol on the left in each mapping pair with the symbol on the right."""
    if isinstance(expr, bool):
        return expr
    return z3.substitute(expr, *mapping) # type: ignore -- z3


#
# Values
#

class Symbol:
    """A class representing a symbolic variable, packaged with associated metadata. Symbols are expected to be uniquely identified by name."""
    def __init__(self, type: CType, name: str, symvar: SymbolicExpression, is_induction_var: bool):
        self.type = type
        self.name = name
        self.symvar = symvar
        self.is_induction_var = is_induction_var

    def __hash__(self):
        return hash(self.name) # names should uniquely identify a symbol.
    
    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        
        # Arguments to the same function are considered positionally equivalent, even if they have different types.
        # Comparing the z3 symvars would just result in an == expression.
        # Therefore, we do not compare either of these.
        assert self.name != other.name or (self.is_induction_var == other.is_induction_var)
        return self.name == other.name
    
    def __repr__(self):
        return f"Symbol({self.type.stubify()}, {self.name}, {self.symvar}, {self.is_induction_var})"

class RODataAddress:
    def __init__(self, literal: StringLiteral):
        self.type = literal.type
        self.name = f"\\str_{literal.value}"
        self.symvar = _z3_pointer_symbol(self.name)
        self.literal = literal

    def __eq__(self, other):
        return isinstance(other, RODataAddress) and self.literal == other.literal

    def __hash__(self):
        return 1 + hash(self.literal)
    
    def __repr__(self):
        return f"&{self.literal}"

def z3repr(object: tuple[CType, str] | Variable | Constant) -> SymbolicExpression: 
    if isinstance(object, Constant):
        match object:
            case IntegerConstant(type=t, value=value): ...
            case FloatConstant(type=t, value=value): ...
            case CharLiteral(type=t, value=value): ...
            case _:
                raise NotImplementedError(f"Support for constant {object} of type {type(object)} is currently not implemented.")
        match t:
            case Integer(size=size): # includes character
                return _z3_integer_value(value, size) # type: ignore
            case Float():
                return _z3_float_value(value, t.size) # type: ignore
    else:
        match object:
            case Variable(type=t, name=name):
                name = "&" + name
                t = Pointer(t)
            case (t, name): ...
        match t:
            case Integer(size=size):
                return _z3_integer_symbol(name, size)
            case Pointer() | Array():
                return _z3_pointer_symbol(name)
            case Float():
                return _z3_float_symbol(name, t.size)
    raise NotImplementedError(f"Can only generate z3 representations of primitive types and pointers, but recieved {t} (from {object})")

class Value:
    def __init__(self, type: CType, expr: SymbolicExpression, field: Field | None = None):
        self.type = type
        self.expr = expr
        self.field = field # only set for Field constants.

    def combine(self, other: "Value", expr: SymbolicExpression, expr_t: CType, prefer_addressability: bool = False) -> "Value":
        if type(other) is not Value:
            return other.combine(self, expr, expr_t, prefer_addressability)
        return Value(expr_t, expr)
    
    def cast(self, new_type) -> "Value":
        if self.type == new_type:
            return self
        else:
            return Value(new_type, cast(self.expr, self.type, new_type))
        
    def substitute(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> "Value":
        return Value(self.type, substitute_z3_expr(self.expr, mapping), self.field)
        
    def __repr__(self):
        return f"({self.type.stubify()}){self.expr}"
    
    @overload
    @staticmethod
    def make(object: StringLiteral) -> "AddressableValue[RODataAddress]": ...
    @staticmethod
    @overload
    def make(object: Constant) -> "Value": ...
    @staticmethod
    @overload
    def make(object: tuple[CType, str]) -> "AddressableValue[Symbol]": ...
    @staticmethod
    @overload
    def make(object: Variable) -> "AddressableValue[Variable]": ...
    @staticmethod
    def make(object: tuple[CType, str] | Variable | Constant) -> "Value":
        if isinstance(object, Field):
            return FieldValue(object)
        elif isinstance(object, StringLiteral):
            base_address = RODataAddress(object)
            return AddressableValue(base_address.type, base_address.symvar, base_address, ())
        symrepr = z3repr(object)
        if isinstance(object, Constant):
            t = typeof(object) # some Constants don't have type attributes; typeof will catch this. 
            return Value(t, symrepr)
        elif isinstance(object, Variable):
            value_type = object.type if isinstance(object.type, Array) else Pointer(object.type)
            return AddressableValue(value_type, symrepr, object, ())
        else:
            t, name = object
            return AddressableValue(t, symrepr, Symbol(t, name, symrepr, False), ())


AddressT = TypeVar("AddressT", Symbol, Variable, RODataAddress)
class AddressableValue(Value,Generic[AddressT]):
    def __init__(self, type: CType, expr: SymbolicExpression, base_address: AddressT, fields: tuple[Field,...]):
        super().__init__(type, expr)
        self.base_address = base_address
        self.fields = fields

    def combine(self, other: Value, expr: SymbolicExpression, expr_t: CType, prefer_addressability: bool = False) -> Value:
        if isinstance(other, AddressableValue) and not prefer_addressability:
            return Value(expr_t, expr)
        if isinstance(other, CompoundValue):
            return other.combine(self, expr, expr_t, prefer_addressability) # will raise an error
        return AddressableValue(expr_t, expr, base_address=self.base_address, fields=self.fields)           

    def cast(self, new_type) -> "AddressableValue[AddressT]":
        if self.type == new_type:
            return self
        else:
            return AddressableValue(new_type, cast(self.expr, self.type, new_type), self.base_address, self.fields)
    
    def substitute(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> "AddressableValue[AddressT]":
        return AddressableValue(self.type, substitute_z3_expr(self.expr, mapping), self.base_address, self.fields)
     
    def compute_offset(self) -> SymbolicExpression:
        """AddressableValues represent a base_address + some offset relative to the base address. This function
        returns just the relative offset without the base address.
        """
        base_symbol = self.base_address.symvar if isinstance(self.base_address, (Symbol, RODataAddress)) else z3repr(self.base_address)
        offset: SymbolicExpression = z3.simplify(self.expr - base_symbol) # type: ignore
        if z3_contains_variable(base_symbol, offset):
            raise ExecutionError(f"Could not remove base address from AddressableValue {self}: {offset}")
        return offset
        
    def __repr__(self):
        desc = f"({self.base_address.name}^{self.type.stubify()})"
        body = str(self.expr) if len(self.fields ) == 0 else f"({self.expr})" + "".join(str(f) for f in self.fields)
        return desc + body
    
Z3_ONE = z3repr(ONE)
Z3_ZERO = z3repr(ZERO)
class ConditionalValue(Value):
    """A wrapper around a z3.BoolRef that converts it to 1 or 0, as the C standard suggests.
    The unwrapped condition is saved separately for use in path conditions.
    """
    def __init__(self, condition: z3.BoolRef | bool):
        super().__init__(INTEGER, z3.If(condition, Z3_ONE, Z3_ZERO))  # type: ignore
        self.condition = condition


class VirtualValue(Value):
    """Represents a value that has no symbolic expression.
    """
    def __init__(self, type: CType, expr_placeholder: SymbolicExpression):
        self.placeholder = expr_placeholder
        super().__init__(type, expr_placeholder)

    @property
    def expr(self) -> SymbolicExpression:
        raise ExecutionError(f"{self.__class__.__name__} has no single symbolic expression.")

    @expr.setter
    def expr(self, value: SymbolicExpression):
        if value is not self.placeholder: # placeholder assignment is allowed during initialization of Value superclass.
            raise ExecutionError(f"Cannot assign to expr attribute of {self.__class__.__name__}: Values are immutable and instances of {self.__class__.__name__} have no single symbolic expression.")
        
    def combine(self, other: "Value", expr: SymbolicExpression, expr_t: CType, prefer_addressability: bool = False) -> Value:
        raise ExecutionError(f"Cannot combine {self.__class__.__name__} {self} with another value.")

    def cast(self, new_type: CType) -> "Value":
        if new_type == self.type:
            return self
        raise SemanticError(f"{self.__class__.__name__} {self} cannot be cast.")
    
    def substitute(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> Value:
        raise ExecutionError(f"Cannot substitute symbols on virtual value of type {self.__class__.__name__}.")

CompoundCTypeT = TypeVar("CompoundCTypeT", Struct, Array)
class CompoundValue(VirtualValue, Generic[CompoundCTypeT]):
    """A value that represents a Struct.

    There are multiple ways to represent a struct in faultless' memory model: as a CompoundValue or as a sequence or writes to an AddressMapping.
    CompoundValues are temporary constructs used to represent structs that exist ephemerally during an expression. Outside of an expression, they
    are stored as writes in an AddressMapping.
    """
    sentinel = z3.Int(f"<CompoundValue>")
    type: CompoundCTypeT

    def __init__(self, compound_t: CompoundCTypeT, offset_values: dict[int, Value]):
        # The superclass constructor expects a variable of type SymbolicExpression but a CompoundValue has no fixed expression type because a z3 expression 
        # can represent only a single scalar at once under faultless' type model. We overwrite expr with a property method that raises an error anyway when 
        # expr is accessed so it doesn't matter what we put here but just in case we an integer variable with a special sentinel name.
        super().__init__(compound_t, CompoundValue.sentinel)
        if isinstance(compound_t, Struct):
            assert len(offset_values) == len(compound_t.members) or type(self) is LazyCompoundValue or type(self) is StringValue
        else:
            assert len(offset_values) == compound_t.nelements or type(self) is LazyCompoundValue or type(self) is StringValue
        self.offset_values = offset_values
    
    def get(self, item: Field | int) -> Value:
        if isinstance(item, Field):
            assert isinstance(self.type, Struct)
            offset = self.type.offsetof(item)
            if offset is None:
                raise SemanticError(f"CompoundValue: cannot find field {item} in {self.type.declaration('')}.")
            return self.offset_values[offset]
        else:
            if item in self.offset_values:
                raise SemanticError(f"CompoundValue: no item at offset {item} in {self.type.declaration('')}.")
            return self.offset_values[item]
    
    def __repr__(self) -> str:
        if isinstance(self.type, Struct):
            parts: list[str] = []
            for member in self.type.members:
                offset = self.type.fieldname2offset[member.name]
                parts.append(f".{member.name}={self.offset_values[offset]}")
            return f"(struct {self.type.name}){{ " + ", ".join(parts) + " }"
        else:
            assert isinstance(self.type, Array)
            parts: list[str] = [str(v) for v in self.offset_values.values()]
            return f"({self.type.stubify().declaration('')}){{" + ", ".join(parts) + "}"
    
    def __iter__(self) -> Iterator[tuple[int, Value]]:
        yield from self.offset_values.items()
    
    def offsets(self):
        return self.offset_values.keys()
    
    def substitute(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> "CompoundValue[CompoundCTypeT]":
        offset_values = {offset: value.substitute(mapping) for offset, value in self.offset_values.items()}
        return CompoundValue(self.type, offset_values)
    
    def decompose(self) -> dict[int, Value]:
        """Convert this CompoundValue into a sequence of non-compound values: the "leaves" of the "tree" created by the CompoundValue."""
        output: dict[int, Value] = {}
        # Guarantee the fields are returned in ascending order.
        for offset, value in sorted(self.offset_values.items(), key=lambda x: x[0]):
            if isinstance(value, CompoundValue):
                for inner_offset, inner_value in value.decompose().items():
                    output[offset + inner_offset] = inner_value
            else:
                output[offset] = value
        return output

    @staticmethod
    def make(object: tuple[CompoundCTypeT, str] | Variable[CompoundCTypeT]) -> "CompoundValue[CompoundCTypeT]":
        if isinstance(object, Variable):
            t = object.type
            base_name = "&" + object.name
        else:
            t, base_name = object
        values = {}
        if isinstance(t, Struct):
            memberstream = ((offset, field.type) for offset, field in t.offset2field.items())
        else:
            assert isinstance(t.element_type, ObjectType)
            element_size = t.element_type.get_size()
            memberstream = ((element_size * offset, t.element_type) for offset in range(t.nelements))
        for offset, element_t in memberstream:
            field_name = f"{base_name}[{offset}]"
            if isinstance(element_t, (Struct, Array)):
                values[offset] = CompoundValue.make((element_t, field_name))
            else:
                values[offset] = Value.make((element_t, field_name))
        return CompoundValue(t, values)

class LazyCompoundValue(CompoundValue[CompoundCTypeT]):
    """A CompoundValue that only is initialized by reading from memory if its fields are accessed.
    
    Often, only a one field of a given struct is actually needed, not all of them. Reads from memory are expensive.
    LazyCompoundValue is behavels like a regular CompoundValue except that it performs those expensive memory reads 
    only if necessary.
    """

    def __init__(self, compound_t: CompoundCTypeT, memory: "AddressMapping[AddressT]", base_address: AddressT, base_offset: "Offset"):
        super().__init__(compound_t, {})
        self.memory = memory
        self.base_address = base_address
        self.base_offset = base_offset

    def is_initialized(self) -> bool:
        assert isinstance(self.type, ObjectType)
        return len(self.offset_values) > 0 or self.type.get_size() == 0
    
    def load(self):
        if not self.is_initialized():
            result = self.memory.read(self.base_address, self.base_offset, self.type)
            assert isinstance(result, CompoundValue)
            self.offset_values = result.offset_values

    def __repr__(self):
        if not self.is_initialized():
            return f"LazyCompoundValue[uninitialized]({self.memory.__class__.__name__}, {self.base_address}, {self.base_offset})"
        return super().__repr__()

    def __iter__(self):
        self.load()
        return super().__iter__()

    def offsets(self):
        self.load()
        return super().offsets()

    def decompose(self) -> dict[int, Value]:
        self.load()
        return super().decompose()
    
    def substitute(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> CompoundValue[CompoundCTypeT]:
        self.load()
        return super().substitute(mapping)
    
class StringValue(CompoundValue[Array]):
    def __init__(self, string: StringLiteral):
        super().__init__(string.type, {})
        self.string = string
        assert isinstance(self.string.type.element_type, Integer)
        self.char_type: Integer = self.string.type.element_type

    @property
    def name(self):
        raise ExecutionError(f"String literal {self} has no name.")

    def get(self, item: int) -> Value:
        if len(self.offset_values) > 0: # no need to re-build the Value if we already have it.
            return super().get(item)
        if item < 0 or item >= len(self.string):
            raise SemanticError(f"Can't access element {item} of string literal {self.string} of length {len(self.string)}")
        return Value.make(self.string.characters[item])
    
    def load(self):
        """It is often not necessary to actually manifest the string literal as a sequence of Values, as many string literals are simply
        arguments to functions. Therefore, we do it lazily.
        """
        if len(self.offset_values) == 0: # String literals always have at least one element: the null terminator.
            char_width = self.char_type.get_size()
            for i, c in enumerate(self.string.characters):
                self.offset_values[i * char_width] = Value.make(c)
    
    def __iter__(self):
        self.load() # will only load if necessary.
        return super().__iter__()

    def __hash__(self):
        return hash(self.string)
    
    def __eq__(self, other):
        return isinstance(other, StringValue) and self.string == other.string

    def __repr__(self):
        return f"(string){self.string}"
    
    def cast(self, new_type: CType) -> "StringValue" | CompoundValue[Array]:
        if isinstance(new_type, Array) and new_type.element_type == self.char_type:
            if len(self.string) <= new_type.nelements:
                return self
            elif len(self.string.characters) == new_type.nelements + 1:
                # C behavior: when a string is copied into an array and there is space for everything
                # except the null terminator, all characters are copied over except the null terminator.
                self.load()
                offset_values = self.offset_values.copy() # preserve immutability
                del offset_values[new_type.nelements * self.char_type.get_size()]
                return CompoundValue(new_type, offset_values)
            else:
                raise SemanticError(f"Cannot fit string {self.string} of length {len(self.string)} into a {new_type}")
        raise SemanticError(f"Cannot cast a string into a {new_type}.")

    def offsets(self):
        char_width = self.char_type.get_size()
        return {i * char_width for i in range(len(self.string))}
    
    def decompose(self) -> dict[int, Value]:
        self.load()
        return super().decompose()

class FieldValue(VirtualValue):
    """A wrapper for a struct/union Field"""
    sentinel = z3.Int("<FieldValue>")

    def __init__(self, field: Field):
        super().__init__(Pointer(Void()), FieldValue.sentinel)
        self.field = field

class VoidValue(VirtualValue):
    """Void represents a lack of a value, so this class is a bit of an oxymoron. However, for the sake
    of cleaner error handling and diagnostics, we model it as a special type of value that throws an 
    error if used.
    """
    sentinel = z3.Int("<FieldValue>")

    def __init__(self):
        super().__init__(Void(), VoidValue.sentinel)
        

class InductiveMemoryAccessMetadata:
    """Contains extra information used to build memory-access queries for self-referential induction variables
    during inductive loop execution.
    """
    def __init__(self, induction_var: SymbolicExpression, base_case: SymbolicExpression | int, update: SymbolicExpression):
        self.induction_var = induction_var
        self.base_case = base_case
        self.update = update

#
# Utility classes and functions
#
class VarNameSource:
    """A utility class that ensures that names are unique by appending unique integers to them.
    """
    def __init__(self):
        self.prefixes: dict[str, int] = {}
    
    def next_name(self, prefix: str):
        if prefix in self.prefixes:
            num = self.prefixes[prefix]
            self.prefixes[prefix] = num + 1
            return prefix + str(num)
        else:
            self.prefixes[prefix] = 0
            return prefix + "0"
VAR_NAME_SOURCE = VarNameSource()

@overload
def copy_to_context(expr: z3.BoolRef | bool, ctx: z3.Context) -> z3.BoolRef: ...
@overload
def copy_to_context(expr: SymbolicExpression | int, ctx: z3.Context) -> SymbolicExpression: ...
def copy_to_context(expr, ctx: z3.Context):
    """Translate a z3 expression into a different context."""
    if not isinstance(expr, z3.ExprRef):
        if isinstance(expr, bool):
            return z3.BoolVal(expr, ctx=ctx)
        elif isinstance(expr, int):
            return z3.IntVal(expr, ctx)
        else:
            raise ValueError(f"Cannot translate value {expr} of type {type(expr)} into the provided context.")
    return expr.translate(ctx) # type: ignore

class VariableMap:
    def __init__(self, mapping: dict[tuple[str, str], z3.BoolRef] | None = None):
        self.mapping = {} if mapping is None else mapping

    def __bool__(self):
        return bool(self.mapping)

    def __len__(self):
        return len(self.mapping)
    
    def __contains__(self, var_pair: tuple[str, str]):
        return var_pair in self.mapping
    
    def __add__(self, other: "VariableMap") -> "VariableMap":
        mapping = self.mapping.copy()
        mapping.update(other.mapping)
        return VariableMap(mapping)
    
    def __setitem__(self, variables: tuple[str, str], expr: z3.BoolRef):
        assert variables not in self.mapping or str(expr) == str(self.mapping[variables]), f"Mismatched equivalence expressions for variables {variables[0]} and {variables[1]}: {self.mapping[variables]} and {expr}"
        self.mapping[variables] = expr

    def __getitem__(self, variables: tuple[str, str]) -> z3.BoolRef:
        return self.mapping[variables]

    def __delitem__(self, variables: tuple[str, str]):
        del self.mapping[variables]

    def __iter__(self) -> Iterable[z3.BoolRef]:
        """Return the equivalence expressions"""
        return iter(self.mapping.values())

    def add_if_compatible(self, left: Symbol, right: Symbol) -> bool:
        """If these two symbols have types with compatible z3 representations, add them to the mapping, 
        casting if necessary, and return True. Otherwise, return False.
        """
        assert isinstance(left, Symbol) and isinstance(right, Symbol)
        compatibility = resolve_to_compatible_z3_repr(left.type, right.type)
        if compatibility:
            if left == right and compatibility is True:
                return True # This is saying, from z3's prespective, that a variable equals itself, which is a tautology.
            elif isinstance(compatibility, CType):
                expr = cast(left.symvar, left.type, compatibility) == cast(right.symvar, right.type, compatibility)
            else:
                expr = left.symvar == right.symvar
            assert expr is not False, f"Equivalent-variable expression {expr} is False, which will cause vacuous equivalence."
            self[(left.name, right.name)] = expr
            return True
        return False

    def update(self, variables: "VariableMap"):
        self.mapping.update(variables.mapping)

    def symbol_mapping(self) -> Iterable[tuple[str, str]]:
        return iter(self.mapping.keys())

    def copy(self) -> "VariableMap":
        return VariableMap(self.mapping.copy())
    
class GlobalAssumptions(VariableMap):
    def __init__(self, lglobals: dict[str, Symbol], rglobals: dict[str, Symbol], memory_symbol_factory: "MemorySymbolFactory"):
        super().__init__()
        self.lglobal_symbols = lglobals
        self.rglobal_symbols = rglobals
        self.memory_symbol_factory = memory_symbol_factory

    def make_equivalence(self, left: str, right: str) -> bool:
        """Attempts to add the equivalence left==right to the mapping, and returns True if it is added.
        (Vacuous equivalences, which are true even without asserting them so like x==x, are not added to 
        the mapping and cause this function to return False).
        """
        original_length = len(self)
        self.add_if_compatible(self.lglobal_symbols[left], self.rglobal_symbols[right])
        return len(self) > original_length

def solve(formula: z3.BoolRef | bool, target_is_sat: bool, solver: z3.Solver | None = None) -> bool:
    target = z3.sat if target_is_sat else z3.unsat
    requires_push = solver is not None
    if solver is None:
        solver = z3.Solver()
    if requires_push:
        solver.push()
    solver.add(formula)
    result = solver.check()
    if result == z3.unknown:
        query_type = "satisfiability" if target_is_sat else "unsatisfiability"
        message = f"Unknown SMT result for {query_type} query with the constraints:\n  " + "\n  ".join(str(c) for c in solver.assertions())
        if requires_push:
            solver.pop()
        raise UnknownSMTResultError(message)
    if requires_push:
        solver.pop()
    return result == target

def satisfiable(formula: z3.BoolRef | bool, solver: z3.Solver | None = None) -> bool:
    return solve(formula, True, solver)

def unsatisfiable(formula: z3.BoolRef | bool, solver: z3.Solver | None = None) -> bool:
    return solve(formula, False, solver)

def valid_expression(expr: bool | z3.BoolRef, solver: z3.Solver | None = None, *var_maps: VariableMap | None) -> bool:
    """Determine whether an expression is valid (always True) if the mapped variables are equivalent to each other.
    """
    expr = z3.Not(expr) # type: ignore
    if var_maps:
        expr = z3.And(*(eq for var_map in var_maps if var_map for eq in var_map), expr) # type: ignore
    return unsatisfiable(expr, solver) # type: ignore

SymExprT = TypeVar("SymExprT", SymbolicExpression, bool | z3.BoolRef)
def equivalent_expressions(left: SymExprT, right: SymExprT, 
                           condition:  z3.BoolRef | bool | None = None, 
                           var_map: VariableMap | None = None,
                           global_assumptions: GlobalAssumptions | None = None,
                           solver: z3.Solver | None = None
                          ) -> bool:
    """Determine if the two expressions left and right are equivalent.
    
    :param condition: These expressions need only be equivalent under this condition.
    :param var_map: The provided variables are set equivalent to each other when proving the expressions equivalent.
    :param global_assumptions: If specified, this function will make the assumption that certain globals are equivalent
        to each other it it helps prove the expression equivalent. If so, it is added to the global_assumptions map.
    :param solver: A z3 solver to use. A default fresh solver will be used instead if none is provided.
    """
    formula = left == right
    if condition is not None:
        formula = z3.Implies(condition, formula)

    is_equivalent = valid_expression(formula, solver, var_map, global_assumptions)

    # If specified, see if we can isolate a pair of global variables that, if set equivalent, would make the
    # entire expression equivalent. Global variables, like parameters, are function inputs, but unlike parameters,
    # they have no prior known equivalence relationship. Instead, we have to determine which are equivalent to
    # each other. We do this by examining the context in which they are used; if globals are used in the same 
    # context, then they are equivalent. In terms of z3 formulas, this means that assuming the globals in two 
    # expressions are equivalent makes them equivalent.
    if not is_equivalent and global_assumptions is not None:
        lglobals = [str(var) for var in z3.z3util.get_vars(left) if str(var).startswith("\\global")]
        rglobals = [str(var) for var in z3.z3util.get_vars(right) if str(var).startswith("\\global")]

        not_present = [
            pair for pair in itertools.product(lglobals, rglobals) 
            if pair not in global_assumptions and (var_map is None or pair not in var_map)
        ]
        # We've isolated a pair of global variables which we haven't previously assumed are equivalent.
        # Let's set them equivalent, if we can, and see if that makes the full expression equivalent.
        if len(not_present) == 1:
            assumption = not_present[0]
            if global_assumptions.make_equivalence(assumption[0], assumption[1]):
                is_equivalent = valid_expression(formula, solver, var_map, global_assumptions)
                if not is_equivalent:
                    del global_assumptions[assumption]
                # At this point, if the expression was shown equivalent, the assumption is recorded in global_assumptions.
    return is_equivalent

def equivalent_values(left: Value, right: Value,
                      condition: z3.BoolRef | bool | None = None,
                      var_map: VariableMap | None = None,
                      global_assumptions: GlobalAssumptions | None = None,
                      permissive_typing: bool = False, 
                      solver: z3.Solver | None = None
                     ) -> bool:
    """Determine whether two Values are equivalent and return True if so, False otherwise.
    """
    if isinstance(left, CompoundValue) and isinstance(right, CompoundValue):
        left_fields = {offset: val for offset, val in left}
        right_fields = {offset: val for offset, val in right}
        if not left_fields.keys() == right_fields.keys():
            return False
        return all(equivalent_values(left_fields[offset], right_fields[offset], condition, var_map, global_assumptions, permissive_typing, solver) for offset in left_fields)
    elif isinstance(left, CompoundValue) or isinstance(right, CompoundValue):
        return False  # a struct is not equivalent to a scalar.
    elif isinstance(left, StringValue) and isinstance(right, StringValue):
        return left.string == right.string
    elif isinstance(left, StringValue) or isinstance(right, StringValue):
        return False
    else: # Equivalence-check two values
        if permissive_typing:
            compatible = resolve_to_compatible_z3_repr(left.type, right.type)
            if not compatible:
                return False
            elif isinstance(compatible, CType):
                left = left.cast(compatible)
                right = right.cast(compatible)
            # else the return value is True and we don't have to do anything.
        elif left.type != right.type:
            return False
        
        return equivalent_expressions(left.expr, right.expr, condition, var_map, global_assumptions, solver)

#
# Path condition
#

class LoopInvariantKey:
    """Serves as a key for loop invariants in ComponentwisePathT, which otherwise would not have a key."""

    def __init__(self, loop_head: "BasicBlock"):
        self.head = loop_head

    def __hash__(self):
        return 10000 + hash(self.head)
    
    def __eq__(self, other):
        return isinstance(other, LoopInvariantKey) and self.head == other.head
    
    def short_desc(self):
        return f"LoopInvariant({self.head.id})"
    
    def __repr__(self):
        return f"LoopInvariant({self.head.id})"


ComponentwisePathT = dict["BasicBlock | NonTautologicalGroup | LoopInvariantKey", tuple[z3.BoolRef | bool, bool]]

class PathCondition:
    """An immutable symbolic path condition with metadata. Wraps the mutable ComponentwisePathT."""

    def __init__(self, components: ComponentwisePathT | None = None):
        """Initalize a path condition. The components argument is for internal use and should not be passed except by methods of this class."""
        # Explanation of "components": it is keyed by the basic block at which the corresponding branch instruction occurs.
        # The values are a tuple of the following: the path condition, and whether this represents the true or false branch path.
        self.components: ComponentwisePathT = {} if components is None else components

    def __bool__(self) -> bool:
        return len(self.components) > 0

    def expr(self) -> z3.BoolRef:
        """Return a z3 symbolic expression representing this PathCondition."""
        if len(self.components) == 0:
            return z3.BoolVal(True)
        elif len(self.components) == 1: # Don't wrap it in an And like below for better display.
            for v in self.components.values():
                condition = v[0]
            return z3.BoolVal(condition) if isinstance(condition, bool) else condition
        else:
            return z3.And(*(v[0] for v in self.components.values())) # type: ignore (due to imprecise z3 type hints.)
        
    def branch(self, block: "BasicBlock", condition: z3.BoolRef | bool, loop_invariant: z3.BoolRef | bool | None = None) -> tuple["PathCondition", "PathCondition"]:
        """Create and return the two path conditions that result from executing a branch instruction. 
        This appends the condition of the branch instruction and its negation to copies of the current path; both are returned.
        """
        true_components = self.components.copy()
        true_components[block] = (condition, True)
        if loop_invariant is not None:
            true_components[LoopInvariantKey(block)] = (loop_invariant, True)
        false_components = self.components.copy()
        false_components[block] = (z3.Not(condition), False) # type: ignore
        return PathCondition(true_components), PathCondition(false_components)
    
    def substitute_variables(self, mapping: list[tuple[SymbolicExpression, SymbolicExpression]]) -> "PathCondition":
        """Create a new path condition with each left-hand expression in mapping replaced by the corresponding right-hand expression.
        """
        # Unfortunately NTGs contain path conditions that need substitution as well.

        def substitute_tree(tree: "PathCondition.MergeTree | int") -> "PathCondition.MergeTree | int":
            if isinstance(tree, int):
                return tree
            return PathCondition.MergeTree(
                substitute_z3_expr(tree.decision, mapping), tree.decision_block,
                substitute_tree(tree.true), substitute_components(tree.true_asserts),
                substitute_tree(tree.false), substitute_components(tree.false_asserts)
            )

        def substitute_component_key(component: "BasicBlock | NonTautologicalGroup | LoopInvariantKey") -> "BasicBlock | NonTautologicalGroup | LoopInvariantKey":
            if isinstance(component, (BasicBlock, LoopInvariantKey)):
                return component
            paths = [substitute_components(path) for path in component.paths]
            new_tree = substitute_tree(component.merge_tree)
            assert isinstance(new_tree, PathCondition.MergeTree)
            return NonTautologicalGroup(new_tree, (paths, component.blocks))

        def substitute_components(components: ComponentwisePathT) -> ComponentwisePathT:
            return {
                substitute_component_key(component): (substitute_z3_expr(condition, mapping), branch)
                for component, (condition, branch) in components.items()
            }

        return PathCondition(substitute_components(self.components))

    def __repr__(self):
        return " and ".join(f"{bb.id if isinstance(bb, BasicBlock) else bb.short_desc()}: {val[1]}" for bb, val in self.components.items())
    
    class MergeTree:
        def __init__(self, decision: z3.BoolRef | bool, decision_block: "BasicBlock",
                           true: "PathCondition.MergeTree | int", true_asserts: ComponentwisePathT, 
                           false: "PathCondition.MergeTree | int", false_asserts: ComponentwisePathT):
            self.decision = decision
            self.decision_block = decision_block
            self.true = true
            self.true_asserts = true_asserts
            self.false = false
            self.false_asserts = false_asserts

        def __repr__(self):
            return f"MergeTree({self.decision}, {self.true}, {self.true_asserts}, {self.false}, {self.false_asserts})"
        
    @staticmethod
    def find_nontautological_tree(tree: "PathCondition.MergeTree | int") -> "PathCondition.MergeTree | int":
        if isinstance(tree, int):
            return -1
        false_branch = PathCondition.find_nontautological_tree(tree.false)
        true_branch = PathCondition.find_nontautological_tree(tree.true)
        if isinstance(false_branch, int) and isinstance(true_branch, int) and not tree.false_asserts and not tree.true_asserts:
            return -1
        else:
            return PathCondition.MergeTree(tree.decision, tree.decision_block, true_branch, tree.true_asserts, false_branch, tree.false_asserts)
    
    @staticmethod
    def merge(in_paths: list["PathCondition"]) -> "tuple[PathCondition, PathCondition.MergeTree]":
        """Combine the paths from multiple predecessors to a basic block. Returns the combined path and a "MergeTree" that reverses
        the branch statements used to build the path conditions.

        :param in_paths: two or more paths to merge
        """
        assert len(in_paths) >= 2, f"Cannot merge fewer than two paths; got {in_paths}"

        # Having a shallow copy of the path component dictionaries is important because they are mutated in the algorithm below.
        mutable_paths: list[ComponentwisePathT] = [p.components.copy() for p in in_paths]
        
        # There are three types of path componets:
        # (1) Background path components. These occur in all path conditions and are all of the same negation (i.e. for 
        #     path condition 'a', all are 'a' or 'not a'). Background path components are not involved in merging, and 
        #     generally hold over this collection of paths.
        # (2) Decision path components: these can be used to partition a set of paths into two nonempty subsets of paths.
        #     The true components and the false components
        # (3) Extra components: these are neither background or decision components. These represent a condition that holds
        #     down a particular branch.
        # Example: If we're merging the following two paths:
        #   {BasicBlockA: (i < n, True), BasicBlockB: (j < k, True), BasicBlockC: (x == 4, True) } and
        #   {BasicBlockA: (i < n, True), BasicBlockB: (z3.Not(j < k), False)}
        # then the path condition component corresponding to BasicBlockA is a background component.
        # the path condition component corresponding to BasicBlockB is a decision component
        # and the path condition component corresponding to BasicBlockC is an extra component

        def build_tree(indexed_paths: list[tuple[int, ComponentwisePathT]]) -> tuple["PathCondition.MergeTree | int", ComponentwisePathT]:
            """Partition the set of paths into two based on some pivot decision path component. All paths where the component
            is true are placed in one partition and all paths where the component is false are placed in another. Return a decision
            tree of this partitioning, where leaves are the indices of uniquely identified paths from the decision tree, and the 
            internal nodes represent partitioning.
            """
            assert len(indexed_paths) > 0, f"Cannot build path merge tree from empty set of paths. Merging called with:" + "\n  ".join(str(p) for p in in_paths)
            
            ### Base case: a path is uniquely identified. Return the index of the path identified and any extra components.
            if len(indexed_paths) == 1:
                return indexed_paths[0]
            
            ### Recursive case: Build a decision tree that can be used to uniquely identify paths.
            # First, find the decision path component which will be used to partition the paths into groups.
            counts = Counter(itertools.chain(*(p[1] for p in indexed_paths)))
            # Update the counts to include the root nodes of NonTautologicalGroups. Track which roots map to which groups.
            # We need only consider the root node because by construction, it is the only node present in all paths that the NTG represents.
            root2ntg: dict[BasicBlock, NonTautologicalGroup] = {}
            for _, path in indexed_paths:
                for component in path:
                    if isinstance(component, NonTautologicalGroup):
                        assert component.root not in path, f"Assumption violation: a basic block cannot be in a path component and in a NonTautologicalGroup in the same path."
                        assert component.root not in root2ntg or root2ntg[component.root] == component, f"Assumption violation: inconsistent NTGs anchored by a given root."
                        root2ntg[component.root] = component
                        counts[component.root] += 1
 
            # counts.update(c.root for p in indexed_paths for c in p[1] if isinstance(c, NonTautologicalGroup))
            frequencies: list[tuple[BasicBlock | NonTautologicalGroup | LoopInvariantKey, int]] = counts.most_common()
            assert frequencies[0][1] == len(indexed_paths), f"Assumption Violation: There is no common decision path component amongst the following paths:\n  " + "\n  ".join(str(p) for p in indexed_paths)
            
            decision_block = None
            background: ComponentwisePathT = {}
            # Find the decision block and any background path components.
            # By definition, each of these occur in all paths under consideration.
            for block, frequency in frequencies:
                assert frequency <= len(indexed_paths)
                if frequency < len(indexed_paths):
                    break

                # These statistics determine if the block is a background or decision block.
                num_true = 0
                num_ntgs = 0
                for _, path in indexed_paths:
                    if block in path:
                        num_true += path[block][1]
                    else:
                        num_ntgs += 1 # If a block is in the frequency count but not in the path, it must be an NTG root.
                consistent_branching_behavior = num_true == len(indexed_paths) or num_true == 0
                consistent_packaging_state = num_ntgs == len(indexed_paths) or num_ntgs == 0

                if consistent_branching_behavior and consistent_packaging_state:
                    # Because we have consistent packaging behavior and we check above that if a block is an NTG root it's not in the path independently
                    # of that NTG, we can conclude that the background element is not the block itself, but the NTG rooted at this block.
                    if block not in indexed_paths[0][1]:
                        assert isinstance(block, BasicBlock)
                        block = root2ntg[block]
                    background[block] = indexed_paths[0][1][block] # by construction all path components featuring this block are either all true or all false.
                else:
                    assert decision_block is None, f"Two possible decision blocks in path merging: {decision_block} and {block}. Input paths:\n " + "\n  ".join(str(p) for p in in_paths)
                    decision_block = block # Partitions the paths into two nonempty subsets
            
            assert decision_block is not None, f"No decision block identified from amongst:\n  " + "\n  ".join(str(p) for p in indexed_paths)
            
            # If the decision condition occurs in a NonTautologicalGroup of paths, then we can't directly split paths
            # based on the pivot, because:
            # (1) The pivot is the root because the root is the only node associated with all paths in the group
            # (2) By construction, the root is associated with at least one false-branch and one true-branch path from the decision node.
            # and because the group can't be put in both the true partition and the false partition.
            # 
            # Therefore, we unpack the group into its individual paths, and append the remaining path components not part
            # of the NTG into that path. Those paths ultimately came from the same branch, so we assign them the same output index.
            if any(decision_block not in indexed_path[1] for indexed_path in indexed_paths):
                # The only counts added to the counter where the decision block is not present occurs when the block is the root of an NTG.
                # That means we have to unpack.
                assert isinstance(decision_block, BasicBlock)
                for i in range(len(indexed_paths)):
                    index, path = indexed_paths[i]
                    if decision_block not in path:
                        ntg = root2ntg[decision_block]
                        del path[ntg] # we want to add all path components to the expanded path except the ntg which we are expanding.
                        for subpath in ntg.paths:
                            expansion = subpath.copy()
                            expansion.update(path)
                            indexed_paths.append((index, expansion))
                        # The path at index is just the original path without the NTG. We appended all the path components to the end of the array.
                        # Therefore, to get rid of that invalid path, we overwrite it by moving the last generated path (valid by construction) to its spot.
                        indexed_paths[i] = indexed_paths.pop()

            # Partition the paths into two groups based on whether the decision path component is true or false.
            decision_condition = None # the decision condition will be found as part of this process
            true_branch: list[tuple[int, ComponentwisePathT]] = []
            false_branch: list[tuple[int, ComponentwisePathT]] = []
            for indexed_path in indexed_paths:
                if indexed_path[1][decision_block][1]:
                    true_branch.append(indexed_path)

                    # The decision condition that we add to the MergeTree should be the true version (i.e. not wrapped in z3.Not(...)).
                    # For the same basic block, the decision is always the same.
                    decision_condition = indexed_path[1][decision_block][0]
                else:
                    false_branch.append(indexed_path)
                del indexed_path[1][decision_block]
                # Also delete any background blocks
                for bg_block in background:
                    del indexed_path[1][bg_block]

            assert decision_condition is not None # Should not be possible because we've manually confirmed this is for a decision block above.
            assert isinstance(decision_block, BasicBlock), f"A {type(decision_block)} cannot be a decision block."

            # Recursively build the path merging decision tree with the two groups computed above.
            return PathCondition.MergeTree(
                decision_condition, decision_block,
                *build_tree(true_branch),
                *build_tree(false_branch)
            ), background

        merge, background = build_tree([(i, c) for i, c in enumerate(mutable_paths)])
        assert not isinstance(merge, int), f"Expected a merge tree with at least one partitioning node with two or more paths:\n. " + "\n. ".join(str(p) for p in in_paths)

        ntt = PathCondition.find_nontautological_tree(merge)
        if isinstance(ntt, PathCondition.MergeTree):
            ntg = NonTautologicalGroup(ntt)
            background[ntg] = (ntg.expr(), True)

        # TODO: add leftover path condition components to the background path components, as we do here.
        condition = PathCondition(background)
        return condition, merge


class NonTautologicalGroup:
    """Represents a collection of two or more paths that in disjunction do not form a tautology."""

    def __init__(self, merge_tree: PathCondition.MergeTree, copy_args: tuple[list[ComponentwisePathT], tuple["BasicBlock",...]] | None = None):
        self.merge_tree = merge_tree
        self.root = self.merge_tree.decision_block
        if copy_args is not None:
            self.paths, self.blocks = copy_args
        else:
            self.paths: list[ComponentwisePathT] = []
            blocks: set[BasicBlock] = set()
            self._init(self.merge_tree, [], [], blocks) # build paths and blocks
            self.blocks: tuple[BasicBlock,...] = tuple(sorted(blocks, key=lambda x: x.id))
    
    def _init(self, tree: PathCondition.MergeTree | int, conditions: list[tuple["BasicBlock | NonTautologicalGroup | LoopInvariantKey", z3.BoolRef | bool]], negated: list[bool], blocks: set["BasicBlock"]):
        if isinstance(tree, int):
            path = { bb: (cond, neg) for (bb, cond), neg in zip(conditions, negated) }
            self.paths.append(path) # type: ignore
        else:
            blocks.add(tree.decision_block)

            conditions.append((tree.decision_block, tree.decision))
            negated.append(True)
            self._extend_asserts(tree.true_asserts, conditions, negated, blocks)
            self._init(tree.true, conditions, negated, blocks)
            self._remove_asserts(tree.true_asserts, conditions, negated)
            negated.pop()
            conditions.pop()

            conditions.append((tree.decision_block, z3.Not(tree.decision))) # type: ignore -- z3 typing
            negated.append(False)
            self._extend_asserts(tree.false_asserts, conditions, negated, blocks)
            self._init(tree.false, conditions, negated, blocks)
            self._remove_asserts(tree.false_asserts, conditions, negated)
            negated.pop()
            conditions.pop()
            

    def _extend_asserts(self, asserts: ComponentwisePathT, conditions: list[tuple["BasicBlock | NonTautologicalGroup | LoopInvariantKey", z3.BoolRef | bool]], negated: list[bool], blocks: set["BasicBlock"]):
        for cond_id, (cond, neg) in asserts.items():
            if isinstance(cond_id, NonTautologicalGroup):
                blocks.update(cond_id.blocks)
            elif isinstance(cond_id, BasicBlock):
                blocks.add(cond_id)
            else:
                # Loop invariants are associated with a given block, but that block should already be in blocks.
                assert cond_id.head in blocks, f"Found {cond_id} but the corresponding loop condition is not a parent in the NTG: {blocks}"
            conditions.append((cond_id, cond))
            negated.append(neg)

    def _remove_asserts(self, asserts: ComponentwisePathT, conditions: list[tuple["BasicBlock | NonTautologicalGroup | LoopInvariantKey", z3.BoolRef | bool]], negated: list[bool]):
        for _ in range(len(asserts)):
            conditions.pop()
            negated.pop()

    def short_desc(self) -> str:
        return f"NTG(" + ", ".join(str(b.id) for b in self.blocks) + ")"

    def expr(self) -> z3.BoolRef | bool:
        """Return a z3 symbolic expression representing this collection of paths."""
        return z3.Or(*(PathCondition(p).expr() for p in self.paths)) # type: ignore

    def with_index(self, index: int) -> "NonTautologicalGroup":
        def build_tree(tree: PathCondition.MergeTree | int):
            if isinstance(tree, int):
                return index
            else:
                return PathCondition.MergeTree(
                    tree.decision, tree.decision_block,
                    build_tree(tree.true), tree.true_asserts.copy(),
                    build_tree(tree.false), tree.false_asserts.copy()
                )
        new_tree = build_tree(self.merge_tree)
        assert isinstance(new_tree, PathCondition.MergeTree)
        return NonTautologicalGroup(new_tree, (self.paths.copy(), self.blocks))

    def __repr__(self):
        return f"NonTautologicalGroup(" + ", ".join(str(b.id) for b in self.blocks) + ")"

    def __hash__(self):
        return hash(self.blocks)
    
    def __eq__(self, other):
        # In a given CFG, a given set of blocks uniquely identifies a path group.
        return isinstance(other, NonTautologicalGroup) and self.blocks == other.blocks


#
# Stack and Heap
#
class Offset:
    def __init__(self, index: SymbolicExpression | int, condition: z3.BoolRef | bool, read_size: int):
        self.index = _z3_pointer_value(index) if isinstance(index, int) else index
        self._condition = condition
        self.read_size = read_size

    def __repr__(self):
        return f"[{self.index}, {self._condition}, {self.read_size}]"
    
    def condition(self, ctx: z3.Context | None, is_read: bool, other_is_inductive: bool):
        if ctx is None:
            return self._condition
        return copy_to_context(self._condition, ctx)

    def adjust_index(self, adjustment: SymbolicExpression | int) -> "Offset":
        """Return a new Offset with the index increased by the adjustment."""
        index: SymbolicExpression = z3.simplify(self.index + adjustment) # type: ignore
        return Offset(index, self._condition, self.read_size)
    
    def refine_condition(self, constraint) -> "Offset":
        """Return a new Offset with the condition conjoined with the additional constraint."""
        return Offset(self.index, z3.And(self._condition, constraint), self.read_size) # type: ignore  (due to imprecise z3 typing)

    def contextualize(self, new_condition) -> "Offset":
        assert type(self) is Offset, f".contextualize() only supported for base offsets but have {type(self)}: {self}" # shouldn't need to support InductiveOffsets
        return Offset(self.index, new_condition, self.read_size)
    
    # Use this if you want to do a change-of-variables on the index.
    # def contextualize(self, var_updates: list[tuple[SymbolicExpression, SymbolicExpression]], new_condition: z3.BoolRef | bool):
    #     """Adapt an offset to a different part of the program."""
    #     assert type(self) is Offset, f".contextualize() only supported for base offsets but have {type(self)}: {self}" # shouldn't need to support InductiveOffsets
    #     index: SymbolicExpression | int
    #     if isinstance(self.index, int):
    #         index = self.index
    #     else:
    #         index = z3.substitute(self.index, *var_updates) # type: ignore
    #     return Offset(index, new_condition, self.read_size)

class InductiveOffset(Offset):
    def __init__(self, index: SymbolicExpression, induction_var: SymbolicExpression, base_case: SymbolicExpression | int, update: SymbolicExpression, condition: z3.BoolRef, read_size: int):
        super().__init__(index, condition, read_size)
        self.induction_var = induction_var
        self.base_case = base_case
        self.update = update

    def __repr__(self):
        return f"[{self.index}, {self.induction_var} in {self.base_case}...{self.update}...{self._condition}, {self.read_size}]"

    def condition(self, ctx: z3.Context | None, is_read: bool, other_is_inductive: bool) -> z3.BoolRef:
        name_suffix = "r" if is_read else "w"
        if ctx is None:
            update = self.update
            induction_var = self.induction_var
            base_case = self.base_case
            loop_bound = self._condition
        else:
            update = copy_to_context(self.update, ctx)
            induction_var = copy_to_context(self.induction_var, ctx)
            base_case = copy_to_context(self.base_case, ctx)
            loop_bound = copy_to_context(self._condition, ctx)
        induction_sort = induction_var.sort()

        update_fn = z3.RecFunction(VAR_NAME_SOURCE.next_name(f"f{name_suffix}"), induction_sort, induction_sort)
        z3.RecAddDefinition(update_fn, (induction_var,), update)

        iter_fn = z3.RecFunction(VAR_NAME_SOURCE.next_name(f"iter{name_suffix}"), z3.IntSort(ctx=ctx), induction_var.sort(), induction_var.sort())
        loopiter = z3.Int('\\loopiter', ctx=ctx)
        z3.RecAddDefinition(iter_fn, (loopiter, induction_var), 
            z3.If(loopiter > 0,
                iter_fn(loopiter - 1, update_fn(induction_var)),
                induction_var
        ))

        eps = z3.Int('\\eps', ctx=ctx)
        valid_iter = induction_var == iter_fn(eps, base_case)

        # condition = z3.And(valid_iter, loop_bound)
        # if not other_is_inductive:
        #     condition = z3.Exists([eps], condition)
        # return condition # type: ignore
        if not other_is_inductive:
            valid_iter = z3.Exists([eps], valid_iter)
        return z3.And(valid_iter, loop_bound) # type: ignore


class MemorySymbolFactory:
    """This class builds symbolic variables for memory reads are not covered by writes."""

    class CacheNode:
        """Stores an item in this class' proven cache."""
        def __init__(self, left_derived: Symbol, right_derived: Symbol, index_eq: z3.BoolRef, varnames: set[str]):
            self.left_derived = left_derived
            self.right_derived = right_derived
            self.index_eq = index_eq
            self.varnames = varnames
            self.prev: MemorySymbolFactory.CacheNode | None = None
            self.next: MemorySymbolFactory.CacheNode | None = None

    class List:
        def __init__(self, head: "MemorySymbolFactory.CacheNode"):
            self.length = 1
            self.head: "MemorySymbolFactory.CacheNode | None" = head
        
        def prepend(self, node: "MemorySymbolFactory.CacheNode"):
            assert node.next is None and node.prev is None
            second = self.head
            if second is not None:
                second.prev = node
            node.next = second
            self.head = node
            self.length += 1

        def delete(self, node: "MemorySymbolFactory.CacheNode"):
            before = node.prev
            after = node.next
            if before is None:
                assert node is self.head
                self.head = after
            else:
                before.next = after
            if after is not None:
                after.prev = before
            node.next = None # we shouldn't need to use this node again but just to be safe
            node.prev = None
            self.length -= 1


    def __init__(self):
        self.symbols: dict[Symbol | Variable | RODataAddress, dict[str, tuple[SymbolicExpression, str, dict[CType, AddressableValue[Symbol] | CompoundValue]]]] = {}
        self.solver = z3.Solver() # All queries are in the same arithmetic query family (everything is in the pointer space) so we use a consistent solver across all queries.

        # These two maps serve as the MemorySymbolFactory's "base proven cache". When two base addresses are proven equivalent, MemorySymbolFactor's derived_symbol_mapping
        # method finds the corresponding indices that are equivalent. Equivalent base addresses and equivalent indices in turn mean that the derived fresh symbolic variables
        # are referring to the same place in memory and therefore are equivalent. Unfortunately, sometimes those indices are defined in terms of variables that have not yet
        # been proven equivalent. For instance, consider
        # int foo(int *x, int n) {                  int bar(int *y, int n) {
        #     for (int i = 0; i < n; ++i)               for (int j = 0; j < n; ++j)
        #         print(x[i]]);                             print(y[j]);
        # }                                         }
        # The print calls are equivalent, but because arguments are aligned at the beginning of proof to provide the base case for the proof, i and j haven't been shown
        # equivalent yet, and so x[i] and y[j] are not shown equivalent at first. It is only after we prove that i == j by showing the corresponding phi instructions equivalent
        # that we can then show x[i] and y[i] (and thus the print calls) equivalent. This cache keeps track of pairs of symbolic variables for which the base address has been
        # proven equivalent but the indicies have not (yet).
        self.left_base_proven_cache: dict[str, MemorySymbolFactory.List] = {}
        self.right_base_proven_cache: dict[str, MemorySymbolFactory.List] = {}
        # set of str(index_eq). Each expression is added to multiple cache lists and we don't want to waste time re-proving already proven (left index == right index) formulas.
        # Another potential issue with re-proving is attribution: we want the variable mapping to be assigned to the var_map for the last necessary pair of values it took to prove 
        # the two index formulas equivalent; if we don't do this, the formula will be trivially re-proven, and its variable mapping attributed to some other arbitrary formula.
        self.proven: set[str] = set()

    def _is_scaled_variable(self, expr: z3.ExprRef) -> bool:
        """Determine if an expression is of the form constant * variable"""
        if expr.decl().kind() not in {z3.Z3_OP_MUL, z3.Z3_OP_BMUL}:
            return False
        
        args = expr.children()
        if len(args) != 2:
            return False

        left, right = args
        numeric_sorts = {z3.Z3_INT_SORT, z3.Z3_REAL_SORT, z3.Z3_BV_SORT, z3.Z3_FLOATING_POINT_SORT}

        return (is_z3_numeric_literal(left) and is_z3_variable(right) and right.sort().kind() in numeric_sorts) or \
               (is_z3_numeric_literal(right) and is_z3_variable(left) and left.sort().kind() in numeric_sorts)

    def fresh_memory_value(self, base_address: AddressT, offset: Offset, read_t: CType) -> AddressableValue[Symbol] | CompoundValue:
        """Get an AddressableValue that describes memory at the location specified in the offset."""
        if isinstance(base_address, RODataAddress):
            raise SemanticError(f"Out-of-bounds read {offset} on string literal {base_address}.")
        assert isinstance(read_t, ObjectType)

        # Get the derived symbol record for this symbol
        if base_address not in self.symbols:
            self.symbols[base_address] = {}
        derived = self.symbols[base_address]

        base_name = base_address.name # both Symbol and Variable have a name field.
        index: SymbolicExpression = _z3_pointer_value(offset.index) if isinstance(offset.index, int) else z3.simplify(offset.index) # type: ignore -- z3 typing
        
        ### Assign a fresh symbolic variable to this read.
        if str(index) in derived:
            _, symbol_name, value_map = derived[str(index)]
        else:
            for saved_index, symbol_name, value_map in derived.values():
                # index mapping exists soley in the pointer space, so we need not worry about types when comparing indices here.
                if unsatisfiable(saved_index != index, self.solver): # We've already allocated a fresh variable that corresponds to this one.
                    break # value_map is now set
            else: # on the for-loop
                ## No name is found. We must create a new one.
                # For easier display and debugging, for simple indices, just show the index directly in a canonical form.
                index_repr = str(index) if is_z3_numeric_literal(index) or is_z3_variable(index) or self._is_scaled_variable(index) else f"\\marg{len(derived)}"
                symbol_name = f"{base_name}[{index_repr}]"
                value_map = {}
                derived[str(index)] = (index, symbol_name, value_map)

        if read_t in value_map:
            return value_map[read_t]
        else:
            if isinstance(read_t, (PrimitiveType, Pointer)):
                value = Value.make((read_t, symbol_name))
                value.base_address.is_induction_var = base_address.is_induction_var if isinstance(base_address, Symbol) else False
            elif isinstance(read_t, Struct):
                value = CompoundValue.make((read_t, symbol_name))
                for subval in value.decompose():
                    assert isinstance(subval, AddressableValue) and isinstance(subval.base_address, Symbol)
                    subval.base_address.is_induction_var = base_address.is_induction_var if isinstance(base_address, Symbol) else False
            else:
                raise NotImplementedError(f"No support yet implemented for allocating fresh variables of type {read_t}")
            
            value_map[read_t] = value
            return value
        
    def get_all_derived_symbols_for(self, base_address: AddressT) -> list[Symbol]:
        """Return every symbol that was derived from this base address."""
        derived_symbols: list[Symbol] = []
        if base_address in self.symbols:
            for _, _, addressable_values in self.symbols[base_address].values():
                for value in addressable_values.values():
                    if isinstance(value, AddressableValue):
                        derived_symbols.append(value.base_address)
                    else:
                        for subval in value.decompose().values():
                            assert isinstance(subval, AddressableValue) and isinstance(subval.base_address, Symbol)
                            derived_symbols.append(subval.base_address)
        return derived_symbols

    def derived_symbol_mapping(self, left: AddressT, right: AddressT, context_map: VariableMap | None, new_map: VariableMap | None = None) -> VariableMap:
        """Return a list of pairs of equivalent symbols derived from left and right."""
        
        # TODO: some symbolic index expressions may be defined in terms of phi variables, call variables, or variables
        # derived from those variables. Thus, in order to prove as many pairs of equivalent derived variables equivalent,
        # we need as context a var_map determining which variables in the left execution are equivalent to which in the
        # right. This is straightforward: add an optional argument containing such a mapping. However, the current prover
        # will be flakey with regards to this because it does not take into account ordering between node pairs it is proving
        # equivalent. For instance, if we have:
        #    x = foo(...)
        #    y = bar(x[1], ...)
        # we'll need the equivalent variable for the symvar x[1] in order to prove bar(...) equivalent to its counterpart.
        # However, the current prover may attempt to prove bar equivalent to its counterpart before proving foo equivalent.
        # The ordering property of dictionaries and the execution order of Execution may guarantee the desired property
        # already, but check.
        
        new_map = VariableMap() if new_map is None else new_map
        if left in self.symbols and right in self.symbols:
            for (lindex, _, lmap), (rindex, _, rmap) in itertools.product(self.symbols[left].values(), self.symbols[right].values()):
                # all indices are in the pointer space so their symbolic expressions should all be the same type.
                equivalent_indices = valid_expression(lindex == rindex, self.solver, context_map, new_map)
                # TODO: handle composite values
                for lvalue, rvalue in itertools.product(lmap.values(), rmap.values()):
                    if isinstance(lvalue, AddressableValue) and isinstance(rvalue, AddressableValue):
                        lbase = lvalue.base_address
                        rbase = rvalue.base_address
                        if equivalent_indices:
                            if new_map.add_if_compatible(lbase, rbase) and isinstance(lbase, Symbol) and isinstance(rbase, Symbol):
                                self.derived_symbol_mapping(lbase, rbase, context_map, new_map) # In turn, these symbols may themselves have derived mappings.
                                self.search_cache(lbase, rbase, context_map, new_map)
                        else:
                            self.add_to_proven_cache(lbase, rbase, lindex, rindex)
        
        return new_map
    
    def add_to_proven_cache(self, left_symbol: Symbol, right_symbol: Symbol, left_index: SymbolicExpression, right_index: SymbolicExpression):
        """Cache a pair of derived symbols along with a formula equating their indices. If these indices are proven equivalent later when provided
        more information, we can pop them out of the cache.
        """
        left_varnames = {str(var) for var in z3.z3util.get_vars(left_index)}
        right_varnames = {str(var) for var in z3.z3util.get_vars(right_index)}
        index_eq = left_index == right_index

        # All of these mean that the two symbols are definitively not equivalent, so there's no point in saving these equalities,
        # trying to prove them later.
        if index_eq is False or len(left_varnames) == 0 or len(right_varnames) == 0:
            return
        
        def _add_to_proven_cache_map(cache_map: dict[str, MemorySymbolFactory.List], this_side_varnames: set[str], other_side_varnames: set[str]):
            for varname in this_side_varnames:
                node = MemorySymbolFactory.CacheNode(left_symbol, right_symbol, index_eq, other_side_varnames)
                if varname in cache_map:
                    cache_map[varname].prepend(node)
                else:
                    cache_map[varname] = MemorySymbolFactory.List(node)

        _add_to_proven_cache_map(self.left_base_proven_cache, left_varnames, right_varnames)
        _add_to_proven_cache_map(self.right_base_proven_cache, right_varnames, left_varnames)

    def search_cache(self, left_symbol: Symbol, right_symbol: Symbol, context_map: VariableMap | None, new_map: VariableMap):
        """Search the cache of symbols with proven equivalent indices.
        """
        if left_symbol.name not in self.left_base_proven_cache or right_symbol.name not in self.right_base_proven_cache:
            return
        left_cache = self.left_base_proven_cache[left_symbol.name]
        right_cache = self.right_base_proven_cache[right_symbol.name]
        
        # Search the shorter list. Each expression appears in the lists of all variables that appear in that expression.
        if left_cache.length <= right_cache.length:
            cache = self.left_base_proven_cache[left_symbol.name]
            symbol = right_symbol
        else:
            cache = self.right_base_proven_cache[right_symbol.name]
            symbol = left_symbol
        
        current: MemorySymbolFactory.CacheNode | None = cache.head
        while current:
            to_delete = None
            if str(current.index_eq) in self.proven:
                to_delete = current
            elif symbol.name in current.varnames:
                # TODO: include global variable assumption logic.
                if valid_expression(current.index_eq, self.solver, context_map, new_map):
                    to_delete = current
                    self.proven.add(str(current.index_eq))
                    if new_map.add_if_compatible(current.left_derived, current.right_derived):
                        self.derived_symbol_mapping(current.left_derived, current.right_derived, context_map, new_map)
            
            current = current.next
            if to_delete:
                cache.delete(to_delete)

    def cache_clear(self):
        """Revert the cache to an empty state"""
        self.left_base_proven_cache = {}
        self.right_base_proven_cache = {}
        self.proven = set()
    
    def __repr__(self):
        lines: list[str] = [
            f"  {name} -> {expr_repr}" 
            for _, derived in self.symbols.items()
            for expr_repr, (_, name, _) in derived.items()
        ]
        return "MemorySymbolMapping:\n" + "\n".join(lines)

class AddressSet(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __hash__(self):
        return id(self)

class Write(AddressSet):
    def __init__(self, offset: Offset, value: Value, history: AddressSet | None):
        self.offset = offset
        self.value = value
        self.history = history

    def __repr__(self):
        return f"({self.offset}:  {self.value}) <- {self.history}"
        

class Join(AddressSet):
    def __init__(self, condition: z3.BoolRef | bool, 
                 true: AddressSet, true_constraints: list[z3.BoolRef | bool], 
                 false: AddressSet, false_constraints: list[z3.BoolRef | bool]):
        self.condition = condition
        self.true = true
        self.true_constraints = true_constraints
        self.false = false
        self.false_constraints = false_constraints

    def __repr__(self):
        return f"Join({self.condition}\n  {self.true}\n  {self.false})\n)"


class AddressMapping(Generic[AddressT]):
    """This is the core class that implements basic read and write operations in the memory model."""

    def __init__(self, mapping: dict[AddressT, AddressSet] | None = None, symbol_factory: MemorySymbolFactory | None = None):
        self.mapping: dict[AddressT, AddressSet] = {} if mapping is None else mapping
        self.symbol_factory = MemorySymbolFactory() if symbol_factory is None else symbol_factory

    def address_space_is_empty(self, base_address: AddressT) -> bool:
        return base_address not in self.mapping

    def write(self, base_address: AddressT, offset: Offset, value_to_store: Value):
        if isinstance(value_to_store, CompoundValue):
            for field_offset, subvalue in value_to_store:
                self.write(base_address, offset.adjust_index(field_offset), subvalue)
        else:
            if base_address not in self.mapping:
                self.mapping[base_address] = Write(offset, value_to_store, None)
            else:
                self.mapping[base_address] = Write(offset, value_to_store, self.mapping[base_address])

    def read(self, base_address: AddressT, query: Offset, read_t: CType, fresh_symbols: list[Symbol] | None = None) -> Value:
        if isinstance(read_t, Struct):
            result: dict[int, Value] = {}
            for field_offset, field_info in read_t.offset2field.items():
                result[field_offset] = self.read(base_address, query.adjust_index(field_offset), field_info.type, fresh_symbols)
            return CompoundValue(read_t, result)
        if isinstance(read_t, Array):
            result: dict[int, Value] = {}
            assert isinstance(read_t.element_type, ObjectType)
            # Local arrays are implemented as pointers to the stack, which avoids a direct read in most cases. However, there are cases when we
            # must read an entire array, including with compound literal expressions and when an array is a struct member and the whole struct is read.
            if MAX_ARRAY_READ_SIZE is not None and read_t.nelements > MAX_ARRAY_READ_SIZE:
                raise UnsupportedFeatureError(f"Reading array of length {read_t.nelements}, larger than configured max length of {MAX_ARRAY_READ_SIZE}")
            elif read_t.nelements >= LARGE_ARRAY_READ_WARN_THRESHOLD:
                warnings.warn(f"Read of array of size {read_t.nelements} may be slow.")
            element_size = read_t.element_type.get_size()
            for arridx in range(read_t.nelements):
                offset = arridx * element_size
                result[offset] = self.read(base_address, query.adjust_index(offset), read_t.element_type, fresh_symbols)
            return CompoundValue(read_t, result)

        if base_address not in self.mapping:
            value = self.symbol_factory.fresh_memory_value(base_address, query, read_t)
            if fresh_symbols is not None:
                assert isinstance(value, AddressableValue) # not always true, TODO: handle composite variables
                fresh_symbols.append(value.base_address)
            return value
        
        ### Search through memory and build a return value. Jointly recursive in find and search_deeper.
        read_solver = z3.Solver()

        def search_deeper(node: Write, query: Offset, refinements: list[z3.BoolRef | bool]) -> Value:
            if node.history is None: # There's no more memory left to search.
                value = self.symbol_factory.fresh_memory_value(base_address, query, read_t)
                if fresh_symbols is not None:
                    assert isinstance(value, AddressableValue) # not always true, TODO: handle composite variables
                    fresh_symbols.append(value.base_address)
                return value
            else:
                return find(node.history, query, refinements)
        
        def find(node: AddressSet, query: Offset, refinements: list[z3.BoolRef | bool]) -> Value:
            if isinstance(node, Write):
                read_is_inductive = isinstance(query, InductiveOffset)
                write_is_inductive = isinstance(node.offset, InductiveOffset)

                #### Check to see if there is some address in the query (memory read) that overlaps with that of the write under consideration.
                # We check this by asking if the following formula is satisfiable:
                # The indices that are involved in the read and write are equivalent.
                # The path conditions that restrict the values of those indices are satisfied.
                solver = read_solver # Re-use the same solver from earlier in this query with the global z3 context.
                overlap_index_eq = node.offset.index == query.index
                overlap_write_condition = node.offset.condition(None, False, read_is_inductive)
                overlap_read_condition = query.condition(None, False, write_is_inductive)
                # Conceptually the refinements are conjoined to the read condition, but it's equivalent to just dump them in the same z3.And.
                overlap = satisfiable(z3.And(overlap_index_eq, overlap_write_condition, overlap_read_condition, *refinements), solver) # type: ignore
                # The formula is not satisfiable means that the data written to memory in this
                # write can't influence the value read from memory specified by the query.
                if not overlap:
                    return search_deeper(node, query, refinements)
                
                #### Check to see if all addresses in the query (memory read) exist in the write under consideration (i.e. the read is a subset of the write.)
                # If this is the case, the following formula should be valid:
                # \forall(phi_r) read_condition is satisfied \implies (\exists(phi_w) (the read and write indices are equal) and (the write condition is satisfied))
                # where phi_r includes the loop-phi variables in the read and phi_w includes all loop-phi variables in the write that aren't also in the read.
                # 
                # Then to check validity with an SMT solver we check that the negation---not(the above formula)---is unsat.
                # not(\forall) is \exists, and SMT solvers assume an implicit existential quantifier, so we just drop the outer quantifier.
                ctx = z3.Context()
                solver = z3.Solver(ctx=ctx) # Overwrite the solver from earlier (in this scope) using the new separate context.
                
                write_index = copy_to_context(node.offset.index, ctx)
                read_index = copy_to_context(query.index, ctx)
                index_eq = write_index == read_index
                read_condition = query.condition(ctx, True, write_is_inductive)
                write_condition = node.offset.condition(ctx, False, read_is_inductive)

                # We want to universally quantify over the loop-phi induction variables that occur in the write but not the read.
                # (If they occur in both, then the variable is constant with respect to this read/write interaction.)
                write_quantification_vars = AddressMapping.get_write_quantification_vars(read_condition, read_index, write_condition, write_index)

                conclusion = z3.Not(z3.And(index_eq, write_condition)) # This formula represents the conclusion of the implication in the non-negated form of the equation.
                if len(write_quantification_vars) > 0: # only wrap in a ForAll if we need to (will throw an exception with an empty list).
                    conclusion = z3.ForAll(write_quantification_vars, conclusion)
                covered_formula = z3.And(read_condition, *(copy_to_context(r, ctx) for r in refinements), conclusion)
                covered = unsatisfiable(covered_formula, solver) # type: ignore

                # Typechecking has already occured, so we don't call assignable here. Rather, we'd like to allow as broad of a type conversion as possible.
                # However, a float-to-integer or pointer cast (or vice versa) is a completely different bitpattern and likely indicates a type-inference bug 
                # (when doing reads within a function) or nonequivalent behavior (when doing reads across execution boundaries in the prover).
                # TODO: maybe make this an InvalidReadError or something like that.
                assert isinstance(read_t, Float) == isinstance(node.value.type, Float), f"Read {node.value} of type {node.value.type} but can't assign this to type {read_t}"
                value = node.value.cast(read_t) # covert the stored type to the read type.
              
                # The formula is valid means that no other prior write to memory can have
                # influenced the value read from memory specified by the query.
                if covered:
                    return value
                else:
                    # The formula used in the z3 If guard differs from the basic overlap formula in several ways. At a high level, they're expressing the same thing:
                    # when this read and write share indices. However, they're applied in different ways.
                    # 1. The loop-induction variables are quantified over in the If guard. The regular overlap condition needs only one satifying assignment,
                    #    while the if-guard should be true for all possible satisfying assignments of the loop induction variable (all relevant loop iterations).
                    # 2. The refinements are not included. Within the context of the z3 If refinements are implicitly provided by the expression's position
                    #    in the z3 AST relative to its parents. For instance, in If(a, If(b, c, d), e), c and d can implicitly be returned only when a is true;
                    #    therefore, it is not necessary to re-include a in the condition of the inner if.
                    # 3. The read condition is not included. The read condition necessarily holds over the entire value by definition. Placing the read condition
                    #    in the if guard provides an interpretation of the value when the read condition is not met, which is doesn't make sense.
                    write_quantification_vars = AddressMapping.get_write_quantification_vars(overlap_read_condition, query.index, overlap_write_condition, node.offset.index)
                    overlap_base_formula = z3.And(overlap_write_condition, overlap_index_eq) # type: ignore
                    if len(write_quantification_vars) > 0: # Only add an exists if we have to
                        overlap_base_formula: z3.BoolRef = z3.Exists(write_quantification_vars, overlap_base_formula) # type: ignore
                    # Because this write does not include all remaining addresses the read can touch, we find values for the remaining addresses by searching deeper in the memory graph.
                    refinements.append(z3.Not(overlap_base_formula)) # type: ignore
                    alternative = search_deeper(node, query, refinements)
                    refinements.pop() # pop off the stack so that we can use the same list for all paths through the memory graph.
                    return Value(read_t, z3.If(overlap_base_formula, value.expr, alternative.expr)) # type: ignore # Due to z3.If's loose typing.
            else:
                assert isinstance(node, Join)
                ctx = z3.Context()
                solver = z3.Solver(ctx=ctx)
                # true_branch and false_branch may be None for one of two reasons:
                # 1. Condition unsatisfiability: The current path condition may preclude going down a branch; that is; the current
                #    path condition implies that the memory state must be from the true branch or must be from the false branch.
                # 2. Shared history: At a branch point, two Writes on different branches can share history (they both point to the same node.) 
                #    If we have already seen this node while exploring another branch, find() returns None.
                refinements.append(node.condition)
                refinements.extend(node.true_constraints)
                true_branch = find(node.true, query, refinements) if satisfiable(z3.And(query.condition(ctx, True, False), *(copy_to_context(r, ctx) for r in refinements)), solver) else None # type: ignore
                if len(node.true_constraints) > 0:
                    del refinements[-len(node.true_constraints):]
                refinements.pop()
                # refinements is now set for the false branch traversal
                refinements.append(z3.Not(node.condition)) # type: ignore
                refinements.extend(node.false_constraints)
                false_branch = find(node.false, query, refinements) if satisfiable(z3.And(query.condition(ctx, True, False), *(copy_to_context(r, ctx) for r in refinements)), solver) else None # type: ignore
                if len(node.false_constraints) > 0:
                    del refinements[-len(node.false_constraints):]
                refinements.pop()
                if true_branch is not None and false_branch is not None:
                    return Value(read_t, z3.If(node.condition, true_branch.expr, false_branch.expr)) # type: ignore # Unfortunately, z3.If's type annotations are very broad.
                elif true_branch is not None:
                    return true_branch
                elif false_branch is not None:
                    return false_branch
                else:
                    raise ExecutionError(f"Memory Model Invariant Error: cannot traverse either the true or false branch of join node with condition {node.condition} with query {query}")
        
        return find(self.mapping[base_address], query, [])
    
    @staticmethod
    def get_write_quantification_vars(read_condition: z3.BoolRef | bool, read_index: SymbolicExpression | int, write_condition: z3.BoolRef | bool, write_index: SymbolicExpression | int):
        """Return the loop-induction variables that occur in the write but not in the read. 
        (If they occur in both, that variable is constant with respect to the memory interaction.)
        """
        read_induction_vars = {str(v) for v in itertools.chain(() if isinstance(read_condition, bool) else z3.z3util.get_vars(read_condition), () if isinstance(read_index, int) else z3.z3util.get_vars(read_index)) if Phi.is_mediloop_symvar(v)}
        # This may return the same variable twice if it exists in both the write condition and write index but this is harmless because z3.Exist ignores duplicate quantified variables.
        return [v for v in itertools.chain(() if isinstance(write_condition, bool) else z3.z3util.get_vars(write_condition), () if isinstance(write_index, int) else z3.z3util.get_vars(write_index)) if Phi.is_mediloop_symvar(v) and v not in read_induction_vars]

class Stack(AddressMapping[Variable]):
    def contains_exactly_initial_write(self, variable: Parameter | GlobalVariable) -> bool:
        """Determine if the stacklet for this variable contains the initial value written to it 
        at the start of execution.
        """
        node = self.mapping[variable]
        def count_components(t: CType) -> int:
            if isinstance(t, Struct):
                return sum(count_components(m.type) for m in t.members)
            elif isinstance(t, Array):
                return count_components(t.element_type) * t.nelements
            else:
                return 1
        # There is one write to the stack for each primitive component of the written value.
        for _ in range(count_components(variable.type)):
            if not isinstance(node, Write):
                return False
            node = node.history
        return node is None

    def copy(self) -> "Stack":
        return Stack(self.mapping.copy(), self.symbol_factory) # symbol factory should not be copied.

T = TypeVar("T")
class Heap(AddressMapping[Symbol]):
    """A Heap inherits the address logic of an AddressMapping, but adds some special logic to handle induction
    on linked data structures that isn't relevant for a Stack. In particular, reads and writes to heap induction
    variables are tracked. Then when the heap is accessed from the variable at the base case of the induction 
    (the case outside the loop), the heap applies those reads and writes to that variable.
    """

    class List(Generic[T]):
        """A simple immutable list class useful for tracking metadata between heap copies."""
        def __init__(self, item: T, next: "Heap.List | None"):
            self.item = item
            self.next = next

        def __iter__(self):
            node = self
            while node is not None:
                yield node.item
                node = node.next
        
        def append(self, item: T) -> "Heap.List":
            return Heap.List(item, self)
        
        def __repr__(self):
            return f"Heap.List(" + " :: ".join(str(item) for item in self) + ")"

    def __init__(self, 
                 mapping: dict[Symbol, AddressSet] | None = None,
                 symbol_factory: MemorySymbolFactory | None = None,
                 derived_ivars: dict[Symbol, Symbol | None] | None = None,
                 inductive_reads: dict[Symbol, "Heap.List[tuple[Offset, CType, Symbol | None]]"] | None = None,
                 inductive_writes: dict[Symbol, "Heap.List[tuple[Offset, Value]]"] | None = None,
                 induction_symbol_binding: dict[Symbol, Symbol] | None = None,
                 base_case_offsets: dict[Symbol, SymbolicExpression] | None = None
                ):
        """
        __init__ should only be called manually with no arguments to initialize a fresh heap.
        Arguments should only be used by the .copy() method.
        
        :param mapping: The core mapping which maps adresses to writes at those addresses.
        :param derived_ivars: A map deriving each induction-associated symbolic variable with the symbolic variable it was derived from.
           This dictionary answers the question "what symbol was the base address when this symbol was allocated?"
        :param inductive_reads: A list of reads performed on induction variables or variables derived from them, as well as any new fresh variables allocated.
        :param inductive_writes: A list of writes performed on induction variabes or variables derived from them, as well as any new fresh variables allocated.
        :param induction_symbol_binding: Maps base case Symbols to the corresonding induction Symbols.
        :param base_case_offsets: The base case may have been at an address offset from the symbolic base address. 
           This dictionary holds the adjustments relative to the base address which when combined acurately represent the address.
        """
        super().__init__(mapping, symbol_factory)
        self.derived_ivars = {} if derived_ivars is None else derived_ivars.copy()
        self.inductive_reads = {} if inductive_reads is None else inductive_reads.copy()
        self.inductive_writes = {} if inductive_writes is None else inductive_writes.copy()
        self.induction_symbol_binding = {} if induction_symbol_binding is None else induction_symbol_binding.copy()
        self.base_case_offsets = {} if base_case_offsets is None else base_case_offsets.copy()

    def copy(self) -> "Heap":
        return Heap(
            self.mapping.copy(),
            self.symbol_factory,
            self.derived_ivars.copy(),
            self.inductive_reads.copy(),
            self.inductive_writes.copy(),
            self.induction_symbol_binding.copy(),
            self.base_case_offsets.copy()
        )
    
    def init_var_induction(self, base_case: Symbol, induction_var: Symbol, base_offset: SymbolicExpression):
        self.derived_ivars[induction_var] = None
        self.induction_symbol_binding[base_case] = induction_var
        self.base_case_offsets[base_case] = base_offset
        # Record base offset here, and apply it in apply_inductive_modifications.

    def write(self, base_address: Symbol, offset: Offset, value_to_store: Value):
        super().write(base_address, offset, value_to_store)

        if base_address.is_induction_var:
            write_record = (offset, value_to_store)
            if base_address in self.inductive_writes:
                self.inductive_writes[base_address] = self.inductive_writes[base_address].append(write_record)
            else:
                self.inductive_writes[base_address] = Heap.List(write_record, None)

    def read(self, base_address: Symbol, query: Offset, read_t: CType, fresh_symbols: list[Symbol] | None = None) -> Value:
        if base_address in self.induction_symbol_binding and valid_expression(query.index == self.base_case_offsets[base_address]):
            self.apply_inductive_modifications(self.induction_symbol_binding[base_address], base_address, query._condition)
            del self.induction_symbol_binding[base_address] # Remove this from the dictionary because we don't need to apply it again

        fresh_vals: list[Symbol] = []
        read_val = super().read(base_address, query, read_t, fresh_vals)

        if fresh_symbols is not None:
            fresh_symbols.extend(fresh_vals)
        
        # Record this read so that it can be applied to the appropriate non-inductive variable if and when necessary.
        if base_address.is_induction_var:
            assert len(set(fresh_vals)) <= 1, f"Heap Invariant Violation: Expected at most one fresh variable per read but found {len(fresh_vals)}: {fresh_vals}"
            read_record = (query, read_t, fresh_vals[0] if len(fresh_vals) == 1 else None)
            if base_address in self.inductive_reads:
                self.inductive_reads[base_address] = self.inductive_reads[base_address].append(read_record)
            else:
                self.inductive_reads[base_address] = Heap.List(read_record, None)
            for fresh_val in fresh_vals:
                self.derived_ivars[fresh_val] = base_address
        return read_val

    def apply_inductive_modifications(self, inductive: Symbol, base: Symbol, condition: z3.BoolRef | bool):
        """Apply the memory modifications applied to the variable 'inductive' during a loop to the variable 'base'
        Can be seen as a applying a function "modify(inductive=base)", where modify represents the modifications made.
        """
        worklist: deque[tuple[Symbol, Symbol]] = deque()
        worklist.append((inductive, base))
        derived: list[tuple[Symbol, Symbol]] = [(inductive, base)]

        # Apply the reads
        while len(worklist) > 0:
            ind, b = worklist.popleft()
            if ind not in self.inductive_reads:
                continue
            for offset, read_t, inductive_fresh in reversed(list(self.inductive_reads[ind])):
                offset = offset.contextualize(condition)
                if base == b:
                    offset = offset.adjust_index(self.base_case_offsets[base])
                freshb: list[Symbol] = []
                super().read(b, offset, read_t, freshb)
                # TODO: This can theoretically happen; fix.
                assert len(freshb) == 0 if inductive_fresh is None else len(set(freshb)) == 1, \
                    f"Heap Invariant Violation: When applying inductive modifications, the base and inductive reads are expected to have the same number of fresh variables: {len(freshb)}: {freshb} vs {int(bool(inductive_fresh))}: {inductive_fresh}."
                if inductive_fresh is not None: # which implies there is a single element in freshb
                    base_fresh = freshb[0]
                    worklist.append((inductive_fresh, base_fresh))
                    derived.append((inductive_fresh, base_fresh))
                    self.induction_symbol_binding[base_fresh] = inductive_fresh
        
        # Apply the writes
        for ind, b in derived:
            if ind not in self.inductive_writes:
                continue
            for offset, val_to_write in reversed(list(self.inductive_writes[ind])):
                offset = offset.contextualize(condition)
                if base == b:
                    offset = offset.adjust_index(self.base_case_offsets[base])
                self.write(b, offset, val_to_write)

class ROData(AddressMapping[RODataAddress]):
    """Represents the read-only data section.
    """
    def __init__(self):
        super().__init__()
        self.values: dict[RODataAddress, StringValue] = {}

    def write(self, base_address: RODataAddress, offset: Offset, value_to_store: Value):
        raise SemanticError(f"Cannot write {value_to_store} to string literal {base_address}: string literals are immutable.")
    
    def read(self, base_address: RODataAddress, query: Offset, read_t: CType, fresh_symbols: list[Symbol] | None = None) -> Value:
        value = self.get_string_value(base_address)
        if z3.is_int_value(query.index):
            return value.get(query.index.as_long()) # type: ignore
        
        if self.address_space_is_empty(base_address): # Only needs to be initialized once.
            # Lazily initialize memory with the contents of the string literal.
            if base_address not in self.mapping:
                for field_offset, character in value:
                    super().write(base_address, Offset(_z3_pointer_value(field_offset), True, base_address.type.nelements), character)
        return super().read(base_address, query, read_t)

    def get_string_value(self, base_address: RODataAddress):
        if base_address not in self.values:
            self.values[base_address] = StringValue(base_address.literal)
        return self.values[base_address]
        

@overload
def merge_memory(address_mappings: list[Stack], merge_tree: PathCondition.MergeTree) -> Stack: ...
@overload
def merge_memory(address_mappings: list[Heap], merge_tree: PathCondition.MergeTree) -> Heap: ...
def merge_memory(address_mappings: Sequence[AddressMapping[AddressT]], merge_tree: PathCondition.MergeTree):
    """Combine two or more memory representations (Stack or Heap) from different paths that meet in the CFG.
    """
    assert len(address_mappings) >= 2, f"Cannot merge an empty set of address mappings"
    def traverse(node: PathCondition.MergeTree, incoming_branches: list[AddressSet]) -> AddressSet:
        true_branch = incoming_branches[node.true] if isinstance(node.true, int) else traverse(node.true, incoming_branches)
        false_branch = incoming_branches[node.false] if isinstance(node.false, int) else traverse(node.false, incoming_branches)
        # If the true and false branch are the same, (which can happen if a variable wan't modified down
        # either path of a given if), then just return that node to simplify the constraints.
        if true_branch is false_branch:
            return true_branch
        else:
            return Join(node.decision,
                        true_branch, [c[0] for c in node.true_asserts.values()], 
                        false_branch, [c[0] for c in node.false_asserts.values()])

    base_addresses = set(address_mappings[0].mapping)
    for address_mapping in address_mappings[1:]:
        base_addresses.update(address_mapping.mapping)

    out_mapping: dict[AddressT, AddressSet] = {}
    for base_address in base_addresses:
        incoming_branches = []
        for address_mapping in address_mappings:
            if base_address in address_mapping.mapping:
                incoming_branches.append(address_mapping.mapping[base_address])
            else:
                # Include a dummy Write with a False path condition, so it can't be read.
                incoming_branches.append(Write(Offset(0, False, -1), Value.make(IntegerConstant(-1, INTEGER)), None))
        
        out_mapping[base_address] = traverse(merge_tree, incoming_branches)
    
    cls = address_mappings[0].__class__
    assert cls is Heap or cls is Stack, f"Attempting to merge unrecognized address mapping type: {cls}"
    assert all(address_mappings[0].symbol_factory is a.symbol_factory for a in address_mappings), f"Inconsistent symbol factory (may produce unsoundness.)"

    if cls is Heap:
        derived_ivars = {}
        inductive_reads = {}
        inductive_writes = {}
        induction_symbol_binding = {}
        base_case_offsets = {}
        for address_mapping in address_mappings:
            derived_ivars.update(address_mapping.derived_ivars) # type: ignore
            inductive_reads.update(address_mapping.inductive_reads) # type: ignore
            inductive_writes.update(address_mapping.inductive_writes) # type: ignore
            induction_symbol_binding.update(address_mapping.induction_symbol_binding) # type: ignore
            base_case_offsets.update(address_mapping.base_case_offsets) # type: ignore
        return Heap(out_mapping, address_mappings[0].symbol_factory, derived_ivars, inductive_reads, inductive_writes, induction_symbol_binding, base_case_offsets) # type: ignore
    else:
        return Stack(out_mapping, address_mappings[0].symbol_factory) # type: ignore


#
# Helper functions and classes for Operations.
#

def typeof(operand: tUnion[Constant, Variable, "SSAInstruction"]) -> CType:
    """Get the C type of an object in the object model that could be used as an operand.
    """
    if isinstance(operand, (Variable, IntegerConstant, FloatConstant, CharLiteral, StringLiteral)):
        return operand.type
    # elif isinstance(obj, SSAInstruction):
    #     pass
    raise SemanticError(f"{type(operand)} objects do not have a C type.")

def integer_promotion(t: Integer) -> Integer:
    """Performs integer promotion, a process by which small integers are temporarily scaled up 
    to larger integers to avoid overflow and other pitfalls of dealing with small integers.
    """
    if t.size < INTEGER.size: # Using size as a proxy for rank
        # An int should always be able to hold all values of a smaller unsigned type.
        return INTEGER
    return t

def arithmetic_type_conversion(left: PrimitiveType, right: PrimitiveType) -> PrimitiveType:
    """Returns the type resulting from when two types are involved in an arithmetic operation.
    """
    # if left == right: # Common case; put this first.
    #     return left
    types = (left, right)
    float_types = tuple(t for t in types if isinstance(t, Float))
    if len(float_types) > 0:
        return max(float_types, key=lambda x: x.size)
    
    # We just have integer types.
    assert isinstance(left, Integer) and isinstance(right, Integer)
    # First do integer promotion.
    left = integer_promotion(left)
    right = integer_promotion(right)
    if type(left) == type(right): # both are signed or both are unsigned but differ in size.
        return max(left, right, key=lambda x: x.size)
    else: # mixed types
        if isinstance(left, SignedInteger):
            assert isinstance(right, UnsignedInteger)
            signed = left
            unsigned = right
        else:
            assert isinstance(right, SignedInteger) and isinstance(left, UnsignedInteger)
            signed = right
            unsigned = left

        # We use size as a proxy for rank here, which is technially incorrect according to the
        # C standard but doesn't result in any semantic difference because the only time differing
        # behavior can result is when two integers of the same rank have the same size.
        if unsigned.size >= signed.size:
            return unsigned
        # For a signed integer to be used, it must be able to fit all values of the unsigned integer.
        # This means that the signed integer must be at least one bit larger
        else:
            return signed
    
def pointer_arithmetic_type_conversion(lt: CType, rt: CType) -> CType:
    # Decay arrays to pointers
    if isinstance(lt, Array):
        lt = Pointer(lt.element_type)
    if isinstance(rt, Array):
        rt = Pointer(rt.element_type)
    ptr_cls = Pointer if isinstance(lt, Pointer) or isinstance(rt, Pointer) else (Pointer, UnknownType)
    if isinstance(lt, ptr_cls):
        pointer_t = lt
        adjustment_t = rt
    else:
        pointer_t = rt
        adjustment_t = lt
    if not isinstance(adjustment_t, (Integer, UnknownType)):
        raise SemanticError(f"Cannot do pointer arithmetic with a {type(adjustment_t)}")
    return pointer_t

def compatible_pointer_types(lt: CType, rt: CType, wild_void: bool = True) -> bool:
    """Determine if two types are compatible. When wild_void is True, Void is allowed
    to match with any other type.
    """
    if isinstance(lt, Pointer):
        l_tgt_type = lt.target_type
    elif isinstance(lt, Array):
        l_tgt_type = lt.element_type
    elif isinstance(lt, UnknownType) and isinstance(rt, (Pointer, Array, UnknownType)):
        return True
    else:
        return False
    if isinstance(rt, Pointer):
        r_tgt_type = rt.target_type
    elif isinstance(rt, Array):
        r_tgt_type = rt.element_type
    elif isinstance(rt, UnknownType) and isinstance(lt, (Pointer, Array)):
        return True
    else:
        return False
    if isinstance(l_tgt_type, UnknownType) or isinstance(r_tgt_type, UnknownType):
        return True
    if wild_void and isinstance(l_tgt_type, Void) or isinstance(r_tgt_type, Void):
        return True
    if is_decompiler_placeholder_type(l_tgt_type) or is_decompiler_placeholder_type(r_tgt_type):
        return True # When used as pointer targets, decompiler placeholder types also function like wildcards
    if isinstance(l_tgt_type, Pointer) and isinstance(r_tgt_type, Pointer):
        return compatible_pointer_types(l_tgt_type, r_tgt_type, wild_void=wild_void)
    # Get as much information about the target types as possible by expanding out incomplete types
    if isinstance(l_tgt_type, (IncompleteStruct, IncompleteUnion)) and l_tgt_type.full_definition is not None:
        l_tgt_type = l_tgt_type.full_definition
    if isinstance(r_tgt_type, (IncompleteStruct, IncompleteUnion)) and r_tgt_type.full_definition is not None:
        r_tgt_type = r_tgt_type.full_definition
    return l_tgt_type == r_tgt_type

def assignable(lhs_t: CType, rhs: Constant | Variable | CType) -> bool:
    """Determine if it is possible to assign a value (rhs) to the type lhs_t.
    Parameter names reflect a simple assignment expression: lhs = rhs.
    """
    rhs_t = rhs if isinstance(rhs, CType) else typeof(rhs)
    if isinstance(lhs_t, UnknownType) or isinstance(rhs_t, UnknownType):
        return True
    return lhs_t == rhs_t or compatible_pointer_types(lhs_t, rhs_t) or \
        (isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType)) or \
        (isinstance(lhs_t, Pointer) and isinstance(rhs, IntegerConstant) and rhs.value == 0) or \
        (isinstance(lhs_t, UnsignedInteger) and lhs_t.name == "_Bool" and isinstance(rhs_t, Pointer)) or \
        (is_decompiler_placeholder_type(lhs_t) and isinstance(rhs_t, (PrimitiveType, Pointer)) and lhs_t.get_size() == rhs_t.get_size()) or \
        (is_decompiler_placeholder_type(rhs_t) and isinstance(lhs_t, (PrimitiveType, Pointer)) and lhs_t.get_size() == rhs_t.get_size())

def resolve(lhs_t: CType, rhs_t: CType, allow_mixed: bool = False, allow_pointers: bool = False) -> CType:
    """For a given arithmetic or relational operation, determine what the output type should be given the input type.

    All relational operators accept primitive types as both operands. Some (+ and -) also accept one pointer and one
    integer (here, denoted 'mixed'). Relational operators allow for comparisons of two pointers.
    """
    if isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType):
        return arithmetic_type_conversion(lhs_t, rhs_t)
    elif allow_mixed and ((isinstance(lhs_t, (Pointer, Array)) and isinstance(rhs_t, Integer)) or (isinstance(lhs_t, Integer) and isinstance(rhs_t, (Pointer, Array)))):
        return pointer_arithmetic_type_conversion(lhs_t, rhs_t)
    elif allow_pointers and compatible_pointer_types(lhs_t, rhs_t):
        if isinstance(lhs_t, Pointer) and isinstance(rhs_t, Pointer):
            if isinstance(lhs_t.target_type, Void):
                return rhs_t
            return lhs_t
        if isinstance(lhs_t, Array): # use the Array's type in case the rhs_t is a void *
            return Pointer(lhs_t.element_type)
        if isinstance(rhs_t, Array):
            return Pointer(rhs_t.element_type)
    raise SemanticError(f"Cannot combine types {lhs_t} and {rhs_t}")

def resolve_to_compatible_z3_repr(lhs_t: CType, rhs_t: CType) -> CType | bool:
    """Attempt to determine a common type for these two types that have the same z3 representation
    so that values in these types can be successfully compared. If the two types already have compatible
    z3 representations, then True is returned. If they can have compatible representations by casting 
    to a common value, that type is returned. If they cannot have a compatible representation, False is
    returned.
    """
    if lhs_t == rhs_t:
        return True
    match (lhs_t, rhs_t):
        case (Integer(), Integer()):
            if Z3_REPR_OPTIONS.integer_repr == "int":
                return True
            return arithmetic_type_conversion(lhs_t, rhs_t)
        case (Float(), Float()):
            if Z3_REPR_OPTIONS.float_repr == "real":
                return True
            return arithmetic_type_conversion(lhs_t, rhs_t)
        case (Pointer(), Pointer()):
            return True
        case (Integer() | Pointer(), Integer() | Pointer()): # must be mixed integer/pointer at this point
            if Z3_REPR_OPTIONS.pointer_repr == Z3_REPR_OPTIONS.integer_repr:
                if Z3_REPR_OPTIONS.integer_repr == "int":
                    return True
                else:
                    lhs_t = lhs_t if isinstance(lhs_t, Integer) else SIZE_T
                    rhs_t = rhs_t if isinstance(rhs_t, Integer) else SIZE_T
                    return arithmetic_type_conversion(lhs_t, rhs_t)
    return False
    
def cast(expr: SymbolicExpression, from_type: CType, to_type: CType) -> SymbolicExpression:
    if from_type == to_type:
        return expr
    if isinstance(from_type, Integer):
        assert isinstance(from_type, SignedInteger) or isinstance(from_type, UnsignedInteger), f"All concrete integer instances must be SignedIntegers or UnsignedIntegers but found {type(from_type)}."
        if isinstance(to_type, Pointer):
            if Z3_REPR_OPTIONS.integer_repr == "bitvec" and Z3_REPR_OPTIONS.pointer_repr == "int":
                return z3.BV2Int(expr, is_signed=isinstance(from_type, SignedInteger))
            if Z3_REPR_OPTIONS.integer_repr == "int" and Z3_REPR_OPTIONS.pointer_repr == "bitvec":
                return z3.Int2BV(expr, _pointer_bitwidth())
            if Z3_REPR_OPTIONS.integer_repr == "bitvec" and Z3_REPR_OPTIONS.pointer_repr == "bitvec":
                assert isinstance(expr, z3.BitVecRef)
                return _bitvec_resize(expr, from_type.size * 8, _pointer_bitwidth(), isinstance(from_type, SignedInteger))
            return expr
        elif isinstance(to_type, Integer):
            if Z3_REPR_OPTIONS.integer_repr == "int":
                return expr
            assert isinstance(expr, z3.BitVecRef), f"{expr}: {type(expr)}; integer repr option: {Z3_REPR_OPTIONS.integer_repr}"
            return _bitvec_resize(expr, from_type.size * 8, to_type.size * 8, isinstance(from_type, SignedInteger))           
        elif isinstance(to_type, Float):
            if Z3_REPR_OPTIONS.float_repr == "real":
                if Z3_REPR_OPTIONS.integer_repr == "bitvec":
                    return z3.ToReal(z3.BV2Int(expr, is_signed=isinstance(from_type, SignedInteger))) # type: ignore
                return z3.ToReal(expr) # type: ignore
            sort = _z3_float_sort(to_type.size)
            if Z3_REPR_OPTIONS.integer_repr == "bitvec":
                assert isinstance(expr, z3.BitVecRef)
                if isinstance(from_type, SignedInteger):
                    return z3.fpSignedToFP(z3.RNE(), expr, sort)
                return z3.fpUnsignedToFP(z3.RNE(), expr, sort)
            return z3.fpToFP(z3.RNE(), z3.ToReal(expr), sort)
        else:
            raise SemanticError(f"Uncastable types: {from_type} to {to_type}")
    elif isinstance(from_type, Float):
        if isinstance(to_type, Float):
            if Z3_REPR_OPTIONS.float_repr == "real":
                return expr
            assert isinstance(expr, z3.FPRef)
            if from_type.size == to_type.size:
                return expr
            return z3.fpToFP(z3.RNE(), expr, _z3_float_sort(to_type.size))
        elif isinstance(to_type, Integer):
            if Z3_REPR_OPTIONS.float_repr == "real":
                assert isinstance(expr, z3.ArithRef)
                truncated = _truncate_real_to_int(expr)
                if Z3_REPR_OPTIONS.integer_repr == "bitvec":
                    return z3.Int2BV(truncated, to_type.size * 8)
                return truncated
            assert isinstance(expr, z3.FPRef)
            bitvec_sort = z3.BitVecSort(to_type.size * 8)
            truncated_bv = z3.fpToSBV(z3.RTZ(), expr, bitvec_sort) if isinstance(to_type, SignedInteger) else z3.fpToUBV(z3.RTZ(), expr, bitvec_sort)
            if Z3_REPR_OPTIONS.integer_repr == "bitvec":
                return truncated_bv
            return z3.BV2Int(truncated_bv, is_signed=isinstance(to_type, SignedInteger))
        else:
            raise SemanticError(f"Uncastable types: {from_type} to {to_type}")
    elif isinstance(from_type, Pointer):
        if isinstance(to_type, Pointer):
            return expr
        elif isinstance(to_type, Integer):
            if Z3_REPR_OPTIONS.pointer_repr == "int" and Z3_REPR_OPTIONS.integer_repr == "bitvec":
                return z3.Int2BV(expr, to_type.size * 8)
            if Z3_REPR_OPTIONS.pointer_repr == "bitvec" and Z3_REPR_OPTIONS.integer_repr == "int":
                return z3.BV2Int(expr, is_signed=False)
            if Z3_REPR_OPTIONS.pointer_repr == "bitvec" and Z3_REPR_OPTIONS.integer_repr == "bitvec":
                assert isinstance(expr, z3.BitVecRef)
                return _bitvec_resize(expr, _pointer_bitwidth(), to_type.size * 8, False)
            return expr
        else:
            raise SemanticError(f"Uncastable types: {from_type} to {to_type}")
    elif isinstance(from_type, Array):
        decayed_type = Pointer(from_type.element_type)
        if isinstance(to_type, (Pointer, Integer)):
            return cast(expr, decayed_type, to_type)
        else:
            raise SemanticError(f"Uncastable types: {from_type} to {to_type}")
    raise NotImplementedError(f"No support for {from_type} to {to_type} casts implemented.")

##############
# Operations #
##############

class Operation:
    def __init__(self):
        raise NotImplementedError("Cannot instantiate abstract Operation object.")
    
    def __eq__(self, other):
        return type(self) == type(other)
    
    def sprint(self, *operands) -> str:
        return " ".join(str(o) for o in operands)
    
    def deduce_type(self, *operands) -> CType | None:
        raise NotImplementedError("Cannot infer type for an abstract Operation object.")
    
    def execute(self, operands: list[Value]):
        raise NotImplementedError(f"Cannot build semantics for abstract Operation object {self}")
    
    def operate(self, left, right) -> z3.BoolRef:
        raise NotImplementedError(f"Abstract {self.__class__.__name__} can't be executed.")

class ExpressionOperation(Operation):
    def deduce_type(self, *operands) -> CType:
        raise NotImplementedError(f"Cannot infer type for an abstract {type(self)} object.")
    
class Infix(ExpressionOperation):
    def __init__(self, operator: str):
        self.operator = operator

    def __str__(self):
        return self.operator
    
    def sprint(self, left, right):
        return f"{left} {self.operator} {right}"
    
# Many of these could be singleton objects but python doesn't have them.
class Addition(Infix):
    def __init__(self):
        super().__init__("+")
    
    def sprint(self, left: tUnion[Constant, Variable, str], right: tUnion[Constant, Variable, str]):
        return f"{left} + {right}"
    
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if isinstance(lt, UnknownType) or isinstance(rt, UnknownType):
            return UnknownType()
        if isinstance(lt, (Pointer, Array)) != isinstance(rt, (Pointer, Array)): # != as xor
            return pointer_arithmetic_type_conversion(lt, rt)
        else:
            if not isinstance(lt, PrimitiveType) or not isinstance(rt, PrimitiveType):
                raise TypeDeductionError(f"Addition between unsupported types: {lt} and {rt}.")
            return arithmetic_type_conversion(lt, rt)
        
    def execute(self, operands: list[Value]):
        l, r = operands
        resolved = resolve(l.type, r.type, allow_mixed=True)
        if isinstance(resolved, Pointer):
            ptr, update = (l, r) if isinstance(l.type, (Pointer, Array)) else (r, l)
            if not isinstance(update.type, Integer):
                raise SemanticError(f"Cannot perform pointer arithmetic with a {update.type} value ({l} + {r})")
            if isinstance(ptr.type, Array):
                ptr = ptr.cast(Pointer(ptr.type.element_type))
            assert isinstance(ptr.type, Pointer)
            # Get the size of the type pointed to so that we can do pointer arithmetic.
            if isinstance(ptr.type.target_type, ObjectType): 
                tgt_type_size = ptr.type.target_type.get_size()
            elif isinstance(ptr.type.target_type, Void):
                tgt_type_size = 1
            else:
                raise ExecutionError(f"Cannot get size for non-object type {ptr.type.target_type} for pointer arithmetic.")
            outexpr = ptr.expr + tgt_type_size * cast(update.expr, update.type, ptr.type)
            return ptr.combine(r, outexpr, resolved, True)
        elif isinstance(resolved, PrimitiveType):
            # This is an assert and not a SemanticError because this should be true after type inference/checking.
            assert isinstance(l.type, PrimitiveType) and isinstance(r.type, PrimitiveType), f"Unexpected argument types for an arithmetic result: {l.type} vs {r.type}."
            outexpr = cast(l.expr, l.type, resolved) + cast(r.expr, r.type, resolved)
            return l.combine(r, outexpr, resolved, False)
        else:
            raise NotImplementedError(f"Addition for {l.type} and {r.type} ({l} and {r}) currently not implemented.")

        
class Subtraction(Infix):
    def __init__(self):
        super().__init__("-")

    @staticmethod
    def _pointer_target_size(pointer_t: Array | Pointer) -> int:
        target_t = pointer_t.element_type if isinstance(pointer_t, Array) else pointer_t.target_type
        if isinstance(target_t, ObjectType):
            return target_t.get_size()
        elif isinstance(target_t, Void):
            return 1
        else:
            raise ExecutionError(f"Cannot get size for non-object type {target_t} for pointer arithmetic.")
    
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if isinstance(lt, UnknownType) or isinstance(rt, UnknownType):
            return UnknownType()
        if isinstance(lt, (Pointer, Array)) and isinstance(rt, Integer):
            return pointer_arithmetic_type_conversion(lt, rt)
        elif isinstance(lt, Array) and isinstance(rt, Array):
            if not lt == rt:
                raise SemanticError(f"Subtracting two pointers is only defined when both pointers point to the same array, but found different pointer types: {lt} and {rt}.")
            return SIZE_T # Actually, should be ptrdiff_t, but this is implementation defined; it is the same as size_t, so we also define it as the same size as SIZE_T.
        elif isinstance(lt, Pointer) and isinstance(rt, Pointer) and lt == rt:
            # Pointer differencing is only defined when both pointes point to the same object and when we just have raw poitners we can't really guarantee that. Otherwise it's undefined behavior.
            raise UnsupportedFeatureError(f"Pointer differencing with no definitive backing array object is unsupported.")
        else:
            if not isinstance(lt, PrimitiveType) or not isinstance(rt, PrimitiveType):
                raise TypeDeductionError(f"Subtraction between unsupported types: {lt} and {rt}.")
            return arithmetic_type_conversion(lt, rt)

    def execute(self, operands: list[Value]):
        l, r = operands
        if isinstance(l.type, Array) and isinstance(r.type, Array):
            if not isinstance(l, AddressableValue) or not isinstance(r, AddressableValue) or l.base_address != r.base_address:
                raise SemanticError(f"Subtracting two pointers is only defined when both pointers point to the same object, but found {l} and {r}.")

            tgt_type_size = self._pointer_target_size(l.type)
            return Value(SIZE_T, cast(l.expr - r.expr, l.type, SIZE_T) / tgt_type_size)
        
        resolved = resolve(l.type, r.type, allow_mixed=True)
        if isinstance(resolved, (Pointer, Array)):
            # As checked in infer_type, pointer subtraction is only valid when the pointer is the left operand
            if isinstance(l.type, Array):
                l = l.cast(Pointer(l.type.element_type))
            assert isinstance(l.type, Pointer) and isinstance(r.type, Integer) # should have been checked by infer_type
            
            tgt_type_size = self._pointer_target_size(l.type)
            outexpr = l.expr - tgt_type_size * cast(r.expr, r.type, l.type)
            return l.combine(r, outexpr, resolved, True)
        else:
            assert isinstance(l.type, PrimitiveType) and isinstance(r.type, PrimitiveType)
            outexpr = cast(l.expr, l.type, resolved) - cast(r.expr, r.type, resolved)
            return l.combine(r, outexpr, resolved, False)
       
class MultiplicativeOperator(Infix): # includes division in the C standard.
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable):
        lt = typeof(left)
        rt = typeof(right)
        if isinstance(lt, UnknownType) or isinstance(rt, UnknownType):
            return UnknownType()
        if not isinstance(lt, PrimitiveType) or not isinstance(rt, PrimitiveType):
            raise TypeDeductionError(f"Unsupported types for {self.operator} operator: {lt} and {rt}.")
        return arithmetic_type_conversion(lt, rt)
    
    def execute(self, operands: list[Value]):
        for op in operands:
            # Is an assert because typechecking should catch this, throwing a semantic error.
            # If we get here, it's due to a bug in the system.
            assert not isinstance(op, AddressableValue) or len(op.fields) == 0, f"Cannot multiply an addressable field offset: {op}"
        l, r = operands
        resolved = resolve(l.type, r.type)
        outexpr = self.operate(cast(l.expr, l.type, resolved), cast(r.expr, r.type, resolved))
        return Value(resolved, outexpr)
    
    def operate(self, left, right) -> SymbolicExpression:
        raise NotImplementedError(f"Abstract MultiplicativeOperator can't be executed.")

class Multiplication(MultiplicativeOperator):
    def __init__(self):
        super().__init__("*")

    def operate(self, left, right) -> SymbolicExpression:
        return left * right
    
class Division(MultiplicativeOperator):
    def __init__(self):
        super().__init__("/")

    def operate(self, left, right) -> SymbolicExpression:
        return left / right
    
class ModulusDivision(MultiplicativeOperator):
    def __init__(self):
        super().__init__("%")
    
    def operate(self, left, right) -> SymbolicExpression:
        return left % right


class BitShift(Infix):
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable):
        lt = typeof(left); rt = typeof(right)
        if isinstance(lt, UnknownType) or isinstance(rt, UnknownType):
            return UnknownType()
        if not isinstance(lt, Integer) or not isinstance(rt, Integer):
            raise TypeDeductionError(f"Arguments to {self.operator} must be integers but got {type(lt)} and {type(rt)}")
        return integer_promotion(lt)

    def execute(self, operands: list[Value]):
        if Z3_REPR_OPTIONS.integer_repr != "bitvec":
            raise ExecutionError(f"Bitshift operations require bitvector integer representation but found {Z3_REPR_OPTIONS.integer_repr}")
        for op in operands:
            assert not isinstance(op, AddressableValue) or len(op.fields) == 0, f"Cannot shift an addressable field offset: {op}"
        l, r = operands
        assert isinstance(l.type, Integer) and isinstance(r.type, Integer), f"Can only bitshift integers."
        resolved = integer_promotion(l.type)
        lc = cast(l.expr, l.type, resolved)
        rc = cast(r.expr, r.type, resolved)
        outexpr = self.operate(lc, rc, resolved)
        return Value(resolved, outexpr)

    def operate(self, left, right, resolved: CType) -> SymbolicExpression:
        raise NotImplementedError(f"Abstract {self.__class__.__name__} can't be executed.")

class LeftShift(BitShift):
    def __init__(self):
        super().__init__("<<")

    def operate(self, left, right, resolved: CType) -> SymbolicExpression:
        return left << right

class RightShift(BitShift):
    def __init__(self):
        super().__init__(">>")

    def operate(self, left, right, resolved: CType) -> SymbolicExpression:
        if isinstance(resolved, UnsignedInteger):
            return z3.LShR(left, right)
        return left >> right


class RelationalOperation(Infix):
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if not (isinstance(lt, (PrimitiveType, UnknownType)) and isinstance(rt, (PrimitiveType, UnknownType))) and \
           not compatible_pointer_types(lt, rt, False):
            raise TypeDeductionError(f"Invalid types for a {self.operator} relational operator: {lt} and {rt}")
        return INTEGER # By definition

class LessThan(RelationalOperation):
    def __init__(self):
        super().__init__("<")
    
    def execute(self, operands: list[Value]):
        l, r = operands
        resolved = resolve(l.type, r.type, allow_pointers=True)
        lc = cast(l.expr, l.type, resolved); rc = cast(r.expr, r.type, resolved)
        if isinstance(resolved, UnsignedInteger) and Z3_REPR_OPTIONS.integer_repr == "bitvec":
            outexpr = z3.ULT(lc, rc)
        else:
            outexpr = lc < rc
        return ConditionalValue(outexpr)
    
class LessThanOrEqualTo(RelationalOperation):
    def __init__(self):
        super().__init__("<=")

    def execute(self, operands: list[Value]):
        l, r = operands
        resolved = resolve(l.type, r.type, allow_pointers=True)
        lc = cast(l.expr, l.type, resolved); rc = cast(r.expr, r.type, resolved)
        if isinstance(resolved, UnsignedInteger) and Z3_REPR_OPTIONS.integer_repr == "bitvec":
            outexpr = z3.ULE(lc, rc)
        else:
            outexpr = lc <= rc
        return ConditionalValue(outexpr)
    
class GreaterThan(RelationalOperation):
    def __init__(self):
        super().__init__(">")

    def execute(self, operands: list[Value]):
        l, r = operands
        resolved = resolve(l.type, r.type, allow_pointers=True)
        lc = cast(l.expr, l.type, resolved); rc = cast(r.expr, r.type, resolved)
        if isinstance(resolved, UnsignedInteger) and Z3_REPR_OPTIONS.integer_repr == "bitvec":
            outexpr = z3.UGT(lc, rc)
        else:
            outexpr = lc > rc
        return ConditionalValue(outexpr)
    
class GreaterThanOrEqualTo(RelationalOperation):
    def __init__(self):
        super().__init__(">=")

    def execute(self, operands: list[Value]):
        l, r = operands
        resolved = resolve(l.type, r.type, allow_pointers=True)
        lc = cast(l.expr, l.type, resolved); rc = cast(r.expr, r.type, resolved)
        if isinstance(resolved, UnsignedInteger) and Z3_REPR_OPTIONS.integer_repr == "bitvec":
            outexpr = z3.UGE(lc, rc)
        else:
            outexpr = lc >= rc
        return ConditionalValue(outexpr)


class EqualityOperation(Infix):
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if not (isinstance(lt, (PrimitiveType, UnknownType)) and isinstance(rt, (PrimitiveType, UnknownType))) and \
           not (isinstance(lt, Pointer) and lt == rt) and not compatible_pointer_types(lt, rt) and \
           not (isinstance(lt, Pointer) and isinstance(right, IntegerConstant) and right.value == 0 or isinstance(rt, Pointer) and isinstance(left, IntegerConstant) and left.value == 0):
            raise TypeDeductionError(f"Invalid operand types for {self.operator} operator: {lt} and {rt}")
        return INTEGER
    
    def execute(self, operands: list[Value]) -> Value:
        l, r = operands
        if isinstance(l.type, Pointer) and isinstance(r.type, Integer) and valid_expression(r.expr == z3_zero(r.type).expr):
            resolved = l.type
        elif isinstance(l.type, Integer) and isinstance(r.type, Pointer) and valid_expression(l.expr == z3_zero(l.type).expr):
            resolved = r.type
        else:
            resolved = resolve(l.type, r.type, allow_pointers=True)
        outexpr = self.operate(cast(l.expr, l.type, resolved), cast(r.expr, r.type, resolved))
        return ConditionalValue(outexpr)
    
class EqualTo(EqualityOperation):
    def __init__(self):
        super().__init__("==")
    
    def operate(self, left, right) -> SymbolicExpression:
        return left == right
    
class NotEqualTo(EqualityOperation):
    def __init__(self):
        super().__init__("!=")

    def operate(self, left, right) -> SymbolicExpression:
        return left != right


class Bitwise(Infix):
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if isinstance(lt, UnknownType) or isinstance(rt, UnknownType):
            return UnknownType()
        if not isinstance(lt, Integer) or not isinstance(rt, Integer):
            raise TypeDeductionError(f"Bitwise {self.operator} requires integer operands but found {lt} and {rt}")
        return arithmetic_type_conversion(lt, rt)

    def execute(self, operands: list[Value]) -> Value:
        if Z3_REPR_OPTIONS.integer_repr != "bitvec":
            raise ExecutionError(f"Bitwise operations require bitvector integer representation but found {Z3_REPR_OPTIONS.integer_repr}")
        for op in operands:
            assert not isinstance(op, AddressableValue) or len(op.fields) == 0, f"Cannot apply bitwise op to an addressable field offset: {op}"
        l, r = operands
        assert isinstance(l.type, Integer) and isinstance(r.type, Integer)
        resolved = arithmetic_type_conversion(l.type, r.type)
        lc = cast(l.expr, l.type, resolved)
        rc = cast(r.expr, r.type, resolved)
        outexpr = self.operate(lc, rc)
        return Value(resolved, outexpr)

    def operate(self, left, right) -> SymbolicExpression:
        raise NotImplementedError(f"Abstract {self.__class__.__name__} can't be executed.")

class BitwiseAnd(Bitwise):
    def __init__(self):
        super().__init__("&")

    def operate(self, left, right) -> SymbolicExpression:
        return left & right

class BitwiseOr(Bitwise):
    def __init__(self):
        super().__init__("|")

    def operate(self, left, right) -> SymbolicExpression:
        return left | right

class BitwiseXOr(Bitwise):
    def __init__(self):
        super().__init__("^")

    def operate(self, left, right) -> SymbolicExpression:
        return left ^ right


class LogicalOperator(Infix):
    def deduce_type(self, left: Constant | Variable, right: Constant | Variable) -> CType:
        lt = typeof(left); rt = typeof(right)
        if not isinstance(lt, (PrimitiveType, Pointer, UnknownType)) or not isinstance(rt, (PrimitiveType, Pointer, UnknownType)):
            raise TypeDeductionError(f"Invalid types for {self.operator} logical operator: {lt} and {rt}")
        return INTEGER
    
    def execute(self, operands: list[Value]) -> ConditionalValue:
        l, r = [truthiness(op) for op in operands]
        return ConditionalValue(self.operate(l, r))

class LogicalAnd(LogicalOperator):
    def __init__(self):
        super().__init__("&&")

    def operate(self, l: z3.BoolRef | bool, r: z3.BoolRef | bool) -> z3.BoolRef:
        return z3.And(l, r) # type: ignore

class LogicalOr(LogicalOperator):
    def __init__(self):
        super().__init__("||")

    def operate(self, l: z3.BoolRef | bool, r: z3.BoolRef | bool) -> z3.BoolRef:
        return z3.Or(l, r) # type: ignore

class ConditionalOperation(ExpressionOperation):
    def __init__(self):
        pass

    def __str__(self):
        return "conditional"
    
    def sprint(self, condition, true_path, false_path):
        return f"{condition} ? {true_path} : {false_path}"
    
    def deduce_type(self, condition, true_path, false_path) -> CType:
        if not isinstance(ct := typeof(condition), (PrimitiveType, Pointer, UnknownType)):
            raise TypeDeductionError(f"Conditional operation must have scalar condition but found {ct}")
        
        tt = typeof(true_path); ft = typeof(false_path)
        if isinstance(tt, UnknownType) or isinstance(ft, UnknownType):
            return UnknownType()
        if isinstance(tt, PrimitiveType) and isinstance(ft, PrimitiveType):
            return arithmetic_type_conversion(tt, ft)
        elif isinstance(tt, (Struct, Union)) and tt == ft:
            return tt
        elif isinstance(tt, Void) and isinstance(ft, Void):
            return tt
        elif isinstance(tt, Pointer) and tt == ft:
            return tt
        elif isinstance(tt, Pointer) and ft == Pointer(Void()) or tt == Pointer(Void()) and isinstance(ft, Pointer):
            return Pointer(Void())
        elif (isinstance(tt, Pointer) and isinstance(false_path, IntegerConstant) and false_path.value == 0 or isinstance(ft, Pointer) and isinstance(true_path, IntegerConstant) and true_path.value == 0):
            return tt if isinstance(tt, Pointer) else ft
        else:
            raise TypeDeductionError(f"Incompatible types for resultant expressions of conditional operator: {tt} and {ft}")

class Cast(ExpressionOperation):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "cast"
    
    def sprint(self, type: CType, value: Constant | Variable):
        return f"({type}){value}"
    
    def deduce_type(self, type: CType, value: Constant | Variable) -> CType:
        if isinstance(type, Void):
            return type
        if not isinstance(type, (PrimitiveType, Pointer, Array)): # arrays decay to pointers
            raise SemanticError(f"Can't cast to non-scalar type {type}")
        if not isinstance(vt := typeof(value), (PrimitiveType, Pointer, Array, UnknownType)):
            raise TypeDeductionError(f"Can't cast value of non-scalar type {vt}")
        return type

    def execute(self, operands) -> Value:
        t, v = operands
        # should always be true if typechecking (e.g. infer_type) is run ahead of time.
        assert isinstance(t, CType)
        assert isinstance(v, Value)
        if isinstance(v, AddressableValue):
            return AddressableValue(t, cast(v.expr, v.type, t), v.base_address, v.fields)
        else:
            assert type(v) is Value or type(v) is ConditionalValue
            return Value(t, cast(v.expr, v.type, t))

class SizeOf(ExpressionOperation):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "sizeof"
    
    def sprint(self, argument: CType):
        return f"sizeof({argument})"
    
    def deduce_type(self, argument: CType | Constant | Variable) -> CType:
        # Technically incomplete types cannot be passed here either but we allow this as part of our deliberate loosening of the C type system.
        if isinstance(ft := argument, FunctionType) or (not isinstance(argument, CType) and isinstance(ft := typeof(argument), FunctionType)):
            raise TypeDeductionError(f"sizeof cannot be applied to function type {ft}")
        return SIZE_T

    def execute(self, operands) -> Value:
        assert len(operands) == 1
        argument = operands[0]
        measured_type = argument if isinstance(argument, CType) else argument.type
        if isinstance(measured_type, FunctionType):
            raise SemanticError(f"sizeof cannot be applied to function type {measured_type}")
        if not isinstance(measured_type, ObjectType):
            raise SemanticError(f"Cannot get size of non-object type {measured_type}.")
        return Value.make(IntegerConstant(measured_type.get_size(), SIZE_T))
    

class Unary(ExpressionOperation):
    def __init__(self, operator):
        self.operator = operator

    def __str__(self) -> str:
        return self.operator
    
    def sprint(self, argument: Constant | Variable):
        return self.operator + str(argument)
    
class UnaryMinus(Unary):
    def __init__(self):
        super().__init__("-")

    def deduce_type(self, argument: Constant | Variable):
        t = typeof(argument)
        if not isinstance(t, (PrimitiveType, UnknownType)):
            raise TypeDeductionError(f"Cannot apply unary - to non-arithmetic operand {t}")
        if isinstance(t, Integer):
            t = integer_promotion(t)
        return t

    def execute(self, operands: list[Value]) -> Value:
        assert len(operands) == 1
        v = operands[0]
        assert not isinstance(v, AddressableValue) or len(v.fields) == 0, f"Cannot apply unary minus to an addressable field offset: {v}"
        if not isinstance(v.type, PrimitiveType):
            raise SemanticError(f"Cannot apply unary - to non-arithmetic operand {v.type}")
        resolved = integer_promotion(v.type) if isinstance(v.type, Integer) else v.type
        return Value(resolved, -cast(v.expr, v.type, resolved))
    
class BitwiseNot(Unary):
    def __init__(self):
        super().__init__("~")

    def deduce_type(self, argument: Constant | Variable):
        t = typeof(argument)
        if isinstance(t, UnknownType):
            return t
        if not isinstance(t, Integer):
            raise TypeDeductionError(f"Cannot apply unary ~ to non-integer operand {t}")
        return integer_promotion(t)

    def execute(self, operands: list[Value]) -> Value:
        if Z3_REPR_OPTIONS.integer_repr != "bitvec":
            raise ExecutionError(f"Bitwise operations require bitvector integer representation but found {Z3_REPR_OPTIONS.integer_repr}")
        assert len(operands) == 1
        v = operands[0]
        assert not isinstance(v, AddressableValue) or len(v.fields) == 0, f"Cannot apply bitwise not to an addressable field offset: {v}"
        if not isinstance(v.type, Integer):
            raise SemanticError(f"Cannot apply unary ~ to non-integer operand {v.type}")
        resolved = integer_promotion(v.type)
        expr = cast(v.expr, v.type, resolved)
        assert isinstance(expr, z3.BitVecRef) # necessary for typing the ~expr, because it only applies to bitvecs.
        return Value(resolved, ~expr)
    
class LogicalNot(Unary):
    def __init__(self):
        super().__init__("!")

    def deduce_type(self, argument: Constant | Variable):
        if not isinstance(t := typeof(argument), (PrimitiveType, Pointer, UnknownType)):
            raise TypeDeductionError(f"Cannot apply unary ! operator to non-scalar operand {t}")
        return INTEGER

    def execute(self, operands: list[Value]) -> ConditionalValue:
        # Is an assert because it should be implicitly checked in infer_type above.
        assert len(operands) == 1, f"Logical not takes exactly one operand but found {len(operands)}"
        operand = operands[0]
        if not isinstance(operand.type, (PrimitiveType, Pointer)):
            raise SemanticError(f"Cannot apply unary ! operator to non-scalar operand {operand.type}")
        return ConditionalValue(z3.Not(truthiness(operand))) # type: ignore -- z3 typing
    
class MemoryOperation:
    """Parent class of those operations which modify memory or otherwise need special access to the stack and heap."""

    @staticmethod
    def type_at_address(addr: AddressableValue):
        match addr.type:
            case Pointer(target_type=ptr_t): ...
            case Array(element_type=ptr_t): ...
            case _:
                raise SemanticError(f"Can only dereference a pointer but found {addr.type}.")
        return ptr_t

    def execute(self, operands: list[tuple[AddressableValue, Value]], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool):
        raise NotImplementedError(f"Cannot execute an abstract MemoryOperation.")
    
class AddressOf(Unary, MemoryOperation):
    def __init__(self):
        super().__init__("&")

    def deduce_type(self, argument: Variable) -> Pointer:
        if isinstance(argument, Constant):
            raise SemanticError(f"Can't take the address of a constant (found {argument}).")
        return Pointer(typeof(argument))
    
    def execute(self, operands: list[Value], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool) -> tuple[AddressableValue, Value]:
        if lval is None: # We allow None as an argument to lval to be consistent with other MemoryOperations.
            raise ExecutionError(f"Executing an address-of operation requires a valid lval but none is provided.")
        return lval, lval
    
class Dereference(Unary, MemoryOperation):
    def __init__(self):
        super().__init__("*")
    
    def deduce_type(self, argument: Variable) -> CType:
        t = typeof(argument)
        if isinstance(t, UnknownType):
            return t
        if not isinstance(t, (Pointer, Array)):
            raise TypeDeductionError(f"Can only dereference a pointer but found {t}.")
        return t.target_type if isinstance(t, Pointer) else t.element_type

    def execute(self, operands: list[Value], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool) -> tuple[AddressableValue, Value]:
        addr, = operands
        if not isinstance(addr, AddressableValue):
            # TODO: attempt decompose expressions to derive an AddressableValue from an eligible expression.
            raise ExecutionError(f"Cannot dereference {addr} because it is not addressable.")
        
        read_t = MemoryOperation.type_at_address(addr)
        if isinstance(read_t, (IncompleteStruct, IncompleteUnion)) and read_t.full_definition is not None:
            read_t = read_t.full_definition
        if not isinstance(read_t, ObjectType):
            raise ExecutionError(f"Cannot read value with non-object type {read_t}")
        read_size = read_t.get_size()
        offset = Offset(addr.compute_offset(), condition, read_size)
        if isinstance(addr.base_address, Symbol):
            read_val = heap.read(addr.base_address, offset, read_t)
        else:
            read_val = stack.read(addr.base_address, offset, read_t)
        return addr, read_val


class Subscript(ExpressionOperation, MemoryOperation):
    def __init__(self):
        pass

    def __str__(self):
        return "[]"
    
    def sprint(self, arr: Constant | Variable, index: Constant | Variable):
        return f"{arr}[{index}]"
    
    def deduce_type(self, arr: Constant | Variable, index: Constant | Variable) -> CType:
        arr_t = typeof(arr); index_t = typeof(index)
        if not isinstance(arr_t, (Array, Pointer, UnknownType)):
            raise TypeDeductionError(f"Cannot perform subscript operation on object of type {arr_t}.")
        if not isinstance(index_t, (Integer, UnknownType)):
            raise TypeDeductionError(f"Subscript index must be an integer type but found {index_t}.")
        if isinstance(arr_t, UnknownType):
            return arr_t
        return arr_t.element_type if isinstance(arr_t, Array) else arr_t.target_type

    def execute(self, operands: list[Value], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool) -> tuple[AddressableValue, Value]:
        base, index = operands
        if not isinstance(base, AddressableValue):
            raise ExecutionError(f"Cannot execute a subscript operation on non-addressable value {base}")

        read_t = MemoryOperation.type_at_address(base)
        if not isinstance(read_t, ObjectType):
            raise ExecutionError(f"Cannot read value with non-object type {read_t}")
        read_size = read_t.get_size()
        member_expr = read_size * cast(index.expr, index.type, Pointer(index.type))
        offset = Offset(base.compute_offset() + member_expr, condition, read_size)
        if isinstance(base.base_address, Symbol):
            read_value = heap.read(base.base_address, offset, read_t)
        else:
            read_value = stack.read(base.base_address, offset, read_t)
        base = AddressableValue(base.type, base.expr + member_expr, base.base_address, base.fields)
        return base, read_value

class MemberAccess(ExpressionOperation, MemoryOperation):
    def __init__(self, indirect: bool):
        self.indirect = indirect

    def __str__(self):
        return "->" if self.indirect else "."
    
    def sprint(self, udt: Constant | Variable, field: Field):
        return f"{udt}{self}{field}"
    
    def deduce_type(self, udt: Constant | Variable, field: Field) -> CType:
        if not isinstance(field, Field):
            raise SemanticError(f"Member of a struct or union must be a field name but found {field}.")
        udt_t = typeof(udt)
        if isinstance(udt_t, UnknownType):
            return udt_t
        if self.indirect:
            if not isinstance(udt_t, Pointer):
                raise TypeDeductionError(f"Cannot apply indirect member access operator -> to non-pointer type {udt_t}")
            udt_t = udt_t.target_type
        if not isinstance(udt_t, (Struct, Union, IncompleteStruct, IncompleteUnion)):
            raise TypeDeductionError(f"Can only access member of struct or union but found {udt_t}.")
        if isinstance(udt_t, (Struct, Union)):
            field_t = udt_t.typeof(field.value)
            if field_t is None:
                raise SemanticError(f"{udt_t.stub} does not have a field named {field}: {udt_t}")
            return field_t
        else:
            return UnknownType()
    
    def execute(self, operands: list[Value], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool) -> tuple[AddressableValue | None, Value]:
        udt_arg, field_val = operands

        assert isinstance(field_val, FieldValue)
        # If we have a CompoundValue, the corresponding value may not even be in memory if it's a literal created in an expression,
        # and even if it is in memory, it would be slower to read it rather than simply accessing it here.
        # In contrast, a LazyCompoundValue must be backed by memory by construction and reading from it would take longer than
        # just reading an individual field.
        if isinstance(udt_arg, CompoundValue) and not isinstance(udt_arg, LazyCompoundValue):
            return lval, udt_arg.get(field_val.field)
        
        base = udt_arg if self.indirect else lval
        if not self.indirect and lval is None:
            raise ExecutionError(f"Executing a local member access (.) on a struct requires a valid lval but none is provided.")

        if not isinstance(base, AddressableValue):
            raise ExecutionError(f"Cannot access field of non-addressable value {base}")
        udt = MemoryOperation.type_at_address(base)
        if isinstance(udt, (IncompleteStruct, IncompleteUnion)) and udt.full_definition is not None:
            udt = udt.full_definition
        if isinstance(udt, (Struct, Union)):
            read_t = udt.typeof(field_val.field.value)
            assert read_t is not None, f"Field {field_val.field} not found." # should have already been checked in infer_type.
            if not isinstance(read_t, ObjectType):
                raise ExecutionError(f"Cannot read value with non-object type {read_t}")
        else:
            read_t = INTEGER

        index_base = base.compute_offset()
        if isinstance(udt, (Struct, IncompleteStruct)):
            # The else condition of this ternary reflects a deprioritized effort to support incomplete structs through the .field attribute of Value
            # FieldValues are now used to represent fields, so this will simply throw an exception when an incomplete struct is passed.
            field_expr = udt.offsetof(field_val.field) if isinstance(udt, Struct) else field_val.expr
            offset = Offset(index_base + field_expr, condition, read_t.get_size())
        else:
            assert isinstance(udt, (Union, IncompleteUnion)), f"Cannot access field of non struct or union {udt}."
            field_expr = 0
            offset = Offset(index_base, condition, read_t.get_size())

        # Only retain fields relative to the current base address.
        fields = (field_val.field,) if self.indirect else base.fields + (field_val.field,)
        
        # When accessed, array struct members decay into pointers to their element types when used in an rvalue context.
        # We don't actually need to read the whole array directly.
        if isinstance(read_t, Array):
            read_addr = AddressableValue(read_t, base.expr + field_expr, base.base_address, fields)
            decayed = AddressableValue(Pointer(read_t.element_type), base.expr + field_expr, base.base_address, fields)
            return read_addr, decayed
        
        if isinstance(base.base_address, Symbol):
            read_val = heap.read(base.base_address, offset, read_t)
        else:
            read_val = stack.read(base.base_address, offset, read_t)
        
        read_addr = AddressableValue(Pointer(read_t), base.expr + field_expr, base.base_address, fields)
        return read_addr, read_val
        
class FunctionCall(ExpressionOperation):
    def __init__(self, fname: tUnion[str, Variable, "SSAInstruction"], ftype: FunctionType | None = None):
        self.fname = fname
        if isinstance(fname, Variable) and not isinstance(fname.type, UnknownType):
            vtype = fname.type
            if not isinstance(vtype, Pointer):
                raise SemanticError(f"Variable {fname} denotes a called function but has non-pointer type {vtype}.")
            _ftype = vtype.target_type
            if not isinstance(_ftype, FunctionType) and not isinstance(_ftype, Void):
                raise SemanticError(f"Variable {fname} denotes a called function but its type is a pointer to {_ftype}.")
            if ftype is not None:
                # This is an assert because if true it reflects a problem with the package implementation's usage of this class, not the C code.
                assert ftype == _ftype, f"Provided mismatching types for function denoted by variable {fname}: {ftype} and {_ftype}"
            if isinstance(_ftype, FunctionType):
                ftype = _ftype
            else:
                ftype = None
        self.ftype: FunctionType | None = ftype

    def __str__(self):
        if self.ftype is not None:
            return self.ftype.declaration(str(self.fname))
        return f"{self.fname}(...)"
    
    def __eq__(self, other):
        # Warning: variables compare based on ID (memory address), which may lead to unintended results when comparing
        # FunctionCalls with variable names (e.g. function pointer calls.)
        return isinstance(other, FunctionCall) and other.fname == self.fname
    
    def sprint(self, *args):
        if isinstance(self.fname, Variable):
            fname = f"(*{self.fname})"
        elif isinstance(self.fname, SSAInstruction):
            fname = str(id(self.fname)) if self.fname.out_repr is None else self.fname.out_repr
            fname = f"(*{fname})"
        else:
            fname = self.fname
        return f"{fname}(" + ", ".join(str(a) for a in args) + ")"
    
    def deduce_type(self, *operands) -> CType:
        if isinstance(self.ftype, FunctionType):
            has_variadic = len(self.ftype.parameters) > 0 and isinstance(self.ftype.parameters[-1][0], FunctionType.VariadicParameter)
            if not has_variadic and len(operands) != len(self.ftype.parameters) or has_variadic and len(operands) < len(self.ftype.parameters) - 1: # -1: It is legal to pass no arguments to a variadic parameter.
                raise SemanticError(f"Improper number of arguments for function of type {self.ftype}: {len(operands)} (" + ", ".join(str(o) for o in operands) +")" )
            for i, ((param_t, _), operand) in enumerate(zip(self.ftype.parameters, operands)):
                if isinstance(param_t, FunctionType.VariadicParameter):
                    break
                if not assignable(param_t, operand):
                    raise TypeDeductionError(f"Incompatible type for argument index {i} to function of type {self.ftype}: {param_t} and {typeof(operand)}.")
            return self.ftype.return_type
        else:
            return UnknownType()
        
    def execute(self, operands: list[Value]):
        # This is because calling functions requires tracking state across an execution run, and because function calls are treated as uninterpreted,
        # the execution behavior is to return a unique per-call symbolic variable. That is very state-based, so this would otherwise be a wrapper around
        # the stateful call-info-tracking object in the interpreter.
        raise ExecutionError(f"Function call behavior is implemented in the interpreter.")
  
class Initializer(ExpressionOperation):
    def __init__(self, type: CType, field_names: list[str] | None = None):
        assert (field_names is not None) == isinstance(type, (Struct, Union, IncompleteStruct, IncompleteUnion)), f"Field names must be provided if and only if the argument is a struct or union but found {type} and {field_names}"
        self.type = type
        self.field_names = field_names

    def __str__(self):
        return "initializer"
    
    def sprint(self, *initial_values: Constant | Variable) -> str:
        if self.field_names is None:
            return f"({self.type})" + "{" + ', '.join(str(val) for val in initial_values) + "}"
        else:
            return f"({self.type})" + "{ " + ", ".join(
                f".{name} = {val}" for name, val in zip(self.field_names, initial_values)
            ) + " }"
        
    def deduce_type(self, *initial_values) -> CType:
        if isinstance(self.type, (Struct, Union)):
            assert self.field_names is not None
            if len(initial_values) != len(self.field_names):
                raise TypeDeductionError(f"Mismatch in number of fields and initial values specified in struct/union initializer: {self.field_names} vs {initial_values}")
            for name, value in zip(self.field_names, initial_values):
                field_t = self.type.typeof(name)
                if field_t is None:
                    raise TypeDeductionError(f"Attempting to initialize a {self.type} but it has no {value} field.")
                elif not assignable(field_t, value):
                    raise TypeDeductionError(f"Cannot initialize struct member \"{field_t} {name}\" with values of type {typeof(value)}.")
        elif isinstance(self.type, Array):
            element_type = self.type.element_type
            for val in initial_values:
                val_t = typeof(val)
                # Initalizers for arrays are slightly more restrictive than assignment in general; you can't assign a pointer to an array.
                # It is also not possible to assign an array to another array, though you can use nested array initializers. This code allows
                # for the use of assigning arrays to arrays, which is more permissive than C allows.
                if not assignable(element_type, val) or (isinstance(element_type, Array) and isinstance(val_t, Pointer)):
                    raise TypeDeductionError(f"Inconsistent operand types in array initializer: {element_type} and {val_t}")
        return self.type

    def execute(self, operands: list[Value]):
        if isinstance(self.type, Struct):
            assert self.field_names is not None # enforced in __init__, reproduced here for the typechecker's benefit.
            # When the items are added to the dictionary as below, the last given item for a field is the value the field takes on, 
            # simulating the behavior of initializer lists.
            initialized = {self.type.offsetof(field): value for field, value in zip(self.field_names, operands)}
            values: dict[int, Value] = {}
            for offset, field in self.type.offset2field.items():
                if offset in initialized:
                    values[offset] = initialized[offset]
                else:
                    values[offset] = z3_zero(field.type) # uninitialized elements are zeroed out
            return CompoundValue(self.type, values)
        if isinstance(self.type, Array):
            if len(operands) > self.type.nelements:
                raise SemanticError(f"Initializer list has too many elements for array of size {self.type.nelements}: {operands}")
            element_size = self.type.get_element_size()
            values = {index * element_size: element.cast(self.type.element_type) for index, element in enumerate(operands)}
            # If necessary, zero out the rest of the array
            if len(operands) < self.type.nelements:
                zero = z3_zero(self.type.element_type)
                for index in range(len(operands), self.type.nelements):
                    values[index * element_size] = zero
            return CompoundValue(self.type, values)
        if isinstance(self.type, (PrimitiveType, Pointer)):
            if len(operands) > 1:
                raise SemanticError(f"An initializer list for a value of type {self.type} must have exactly one element.")
            value = operands[0] if len(operands) == 1 else z3_zero(self.type)
            return value.cast(self.type)
        raise SemanticError(f"Cannot apply initializer list to object of type {self.type}")   


class Store(ExpressionOperation, MemoryOperation):
    def __init__(self):
        pass

    def __str__(self):
        "store"

    def sprint(self, lval, rval):
        return f"store {lval} {rval}"
    
    def deduce_type(self, lval: Variable, rval: Constant | Variable):
        if isinstance(lval, Constant):
            raise TypeDeductionError(f"Cannot store a value in a constant ({lval})")
        lt = typeof(lval)
        if not assignable(lt, rval):
            raise TypeDeductionError(f"Cannot store rval {rval} of type {typeof(rval)} in lval of type {typeof(lval)}")
        return lt # returns the value stored, not the original rval
    
    def execute(self, operands: list[Value], *, lval: AddressableValue | None, stack: Stack, heap: Heap, condition: z3.BoolRef | bool) -> tuple[AddressableValue, Value]:
        _, val_to_store = operands
        if lval is None: # None is allowed as a paramter for consistency with other MemoryOperations.
            raise ExecutionError(f"Cannot store value {val_to_store} because no lval has been provided.")
        if not isinstance(val_to_store.type, ObjectType):
            raise ExecutionError(f"Writing a value requires a size but have non-object type {val_to_store.type}")
        offset = Offset(lval.compute_offset(), condition, val_to_store.type.get_size())
        if isinstance(lval.base_address, Symbol):
            heap.write(lval.base_address, offset, val_to_store)
        else:
            stack.write(lval.base_address, offset, val_to_store)
        return lval, val_to_store

class Phi(ExpressionOperation):
    def __init__(self, variable: Variable, differentiator: str | None = None):
        self.type = variable.type
        self.name = f"\\phi_{variable.name}"
        if differentiator is not None:
            self.name = self.name + differentiator
        self.variable = variable
        self.requires_register = variable.is_temporary
        self.bootstrapped = False
        # Only defined (and therefore set to an int value) if the phi node is a loop-phi
        self.loop_base_case: int | None = None # represents the index of the operand that corresponds to the base case: the value coming from outside the loop.

        postloop_name = f"\\$phi_{variable.name}"
        if differentiator is not None:
            postloop_name += differentiator
        # The compiler generates temporaries for each function callee because at IR generation time the types of instructions
        # are not known. Those temporaries continue to exist, even if they are found to have type void during type inference.
        # However, this should not cause problems during execution because only phis corresponding to either declared or global
        # variables (which type deduction and inference should confirm has a valid object type) or temporaries from control-flow-inducing
        # expressions (&&, ||, or a ternary), which also should have a valid object type.
        # If there's a bug where this is not the case, we should get an AttributeError when trying to access one of the below attributes.
        if not isinstance(self.type, (Void, Struct, Array, FunctionType)):
            self.mediloop_symvar = z3repr((self.type, self.name))
            self.postloop_symvar = z3repr((self.type, postloop_name))
            self.mediloop_value = AddressableValue(self.type, self.mediloop_symvar, Symbol(self.type, self.name, self.mediloop_symvar, True), ())
            self.postloop_value = AddressableValue(self.type, self.postloop_symvar, Symbol(self.type, postloop_name, self.postloop_symvar, False), ())

            # Will be inferred later in .execute():
            self.base_case_value: Value | None = None
            self.induction_tuple: InductiveMemoryAccessMetadata | None = None
            self.path_condition: list[z3.BoolRef | bool] = []
        else:
            assert variable.is_stack_allocated == False, f"Cannot create phi-vars for stack-allocated variable of type {variable.type}"

    @staticmethod
    def is_mediloop_symvar(var: SymbolicExpression) -> bool:
        """Determine if the argument is a loop-phi variable active during the loop (a mediloop var).
        Requires the argument to be a variable.
        """
        assert z3.is_const(var) and var.decl().kind() == z3.Z3_OP_UNINTERPRETED, f"{var} is not a variable."
        return str(var).startswith("\\phi_")

    def __str__(self):
        return "phi"
    
    def sprint(self, *operands):
        return f"phi(" + ', '.join(str(o) for o in operands) + ")"
    
    def deduce_type(self, *operands: Constant | Variable) -> CType:
        if len(operands) == 0:
            raise SemanticError(f"Phi operation with no operands is invalid.")
        for operand in operands:
            if isinstance(operand, Variable):
                var_t = typeof(operand)
                if not isinstance(var_t, UnknownType):
                    break
        # The variable itself probably has an unknown type. See if we can
        # infer a type from the constants in the arguments, if any.
        if isinstance(var_t, UnknownType):
            for operand in operands:
                if isinstance(operand, Constant):
                    var_t = typeof(operand)
                    break
            if isinstance(var_t, UnknownType):
                return var_t
        for operand in operands:
            t = typeof(operand)
            # Rationale for this test: a phi instruction represents the same variable from different paths,
            # and a variable should always have the same type as itself. That being said, real code is often
            # very loose with the type of numeric constants and relies on the compiler to automatically adjust 
            # the constant's type to the variable's type. Thus, we're more flexible here on that front, returning
            # the type of the variable.
            if var_t != t and not isinstance(t, UnknownType) and \
               not (isinstance(operand, IntegerConstant) and isinstance(var_t, Integer)) and \
               not (isinstance(operand, FloatConstant) and isinstance(var_t, Float)):
                raise TypeDeductionError(f"Inconsistent types in phi operation: {var_t} and {t}")
        return var_t
    
    @overload
    def execute(self, operands: list[Value], *, first_exec: Literal[True],  block_path_condition: list[z3.BoolRef | bool], heap: Heap) -> tuple[AddressableValue, InductiveMemoryAccessMetadata | None]: ...
    @overload
    def execute(self, operands: list[Value], *, first_exec: Literal[False], block_path_condition: list[z3.BoolRef | bool], heap: Heap) -> tuple[AddressableValue, None]: ...
    def execute(self, operands: list[Value], *, first_exec: bool, block_path_condition: list[z3.BoolRef | bool], heap: Heap):
        """Execute a phi instruction. Phi instructions handle most of the logic around induction.
        """
        assert self.loop_base_case is not None # Non-loop phi instructions are implicitly handled by the stack.

        assert len(operands) == 1, f"A loop-phi instruction should only be called with one operand."
        operand = operands[0]
        if not self.bootstrapped: # first pass
            if first_exec:
                assert self.base_case_value is None, f"Base case value should be None on the first execution. (bootstrapped={self.bootstrapped})"
                self.base_case_value = operand
                return self.mediloop_value, None
            else:
                # Compute and store the necessary metadata for each type of node
                
                # Comparing the base case to itself will undermine the comparison; remove it from the operands list.
                base_case = self.base_case_value
                assert base_case is not None, f"Must call Phi.execute() with first_exec=True once before calling it with first_exec=False"

                if z3_contains_variable(self.mediloop_symvar, operand.expr): # The operand is defined in terms of the base case.
                    # The update is the coefficient because it is repeatedly added to the looping variable. Loops of the form
                    #   for (int i = b; i < n; i += u)
                    # can be represented by the equation i = u*phi + b, where phi represents the loop iteration. 
                    # Loops that are of this form pass the if-check below.
                    coefficient: SymbolicExpression = z3.simplify(operand.expr - self.mediloop_symvar) # type: ignore
                    if z3_contains_variable(self.mediloop_symvar, coefficient):
                        # This means update is NOT of the form phi + u where phi is the induction variable.
                        # The usually more efficient affine form computed below therefore cannot be used, and we
                        # fall back on the more general InductiveOffset.
                        self.induction_tuple = InductiveMemoryAccessMetadata(self.mediloop_symvar, base_case.expr, operand.expr)
                    else:
                        # Because the coefficient can be any symbolic expression, it may not always be the case that it
                        # is always increasing or decreasing (e.g. if it is an unconstrained symbolic variable). Therefore,
                        # we separately check both cases.

                        # z3.Not for validity query
                        increasing = unsatisfiable(z3.Not(coefficient > 0)) # type: ignore -- due to very broad z3 typing
                        decreasing = unsatisfiable(z3.Not(coefficient < 0)) # type: ignore -- due to very broad z3 typing
                        assert not (increasing and decreasing), f"Invariant issue: {operand} is both increasing and decreasing."

                        # Instead of storing these in an induction tuple which can be used for building memory model queries,
                        # we add this information directly into the path. Induction tuple information is ultimately added to the
                        # path conditions in memory model queries, but these affine update constraints are so cheap to evaluate
                        # that it makes sense to just add them to the path condition so we get more precise path decisions for free.
                        # Also, by adding them to the path condition, they implicilty become a part of memory model queries.
                        if increasing:
                            self.path_condition.append(self.mediloop_symvar >= base_case.expr)
                        elif decreasing:
                            self.path_condition.append(self.mediloop_symvar <= base_case.expr)

                        # It is not sufficient to set a lower and upper bound for the looping variable; we further want to constrain the 
                        # looping variable to values it could actually take during the loop. For instance, for the loop
                        #   for (int i = 1; i < n; i += 2) { ... }
                        # we want to constraint i to be odd (i.e. i % 2 == 1). If the base_case >= coefficient, e.g. 3 >= 2 in 
                        #   for (int i = 3; i < n; i += 2) { ... }
                        # then this "step" constraint is automatically UNSAT by itself, dragging the rest of the path constraint with it.
                        # Thus, it is actually necessary to constrain the remainders of both the path condition and the base case to be
                        # equivalent to each other.
                        if isinstance(self.type, Integer): # modulus division only applies to integers
                            self.path_condition.append(self.mediloop_symvar % coefficient == base_case.expr % coefficient)
                self.bootstrapped = True
                self.base_case_value = None
                return self.postloop_value, None
        else: # The main execution, not the bootstrapping execution
            if first_exec:
                block_path_condition.extend(self.path_condition)
                # Only the base case is passed in on the first execution (enforced above)
                assert self.base_case_value is None, f"Base case value should be None on the first execution. (bootstrapped={self.bootstrapped})"
                self.base_case_value = base_case = operand
                if isinstance(base_case, AddressableValue) and isinstance(base_case.base_address, Symbol):
                    # TODO: check that handling base addresses of type Variable is not necessary.
                    heap.init_var_induction(base_case.base_address, self.mediloop_value.base_address, base_case.compute_offset())
                return self.mediloop_value, self.induction_tuple
            else:
                return self.postloop_value, None

class Copy(ExpressionOperation):
    def __init__(self):
        pass

    def __str__(self):
        return "copy"
    
    def sprint(self, arg):
        return str(arg) # the assignment part is printed by Instruction, not Operation.
    
    def deduce_type(self, operand: Constant | Variable) -> CType:
        return typeof(operand)
    
    def execute(self, operands: list[Value]):
        assert len(operands) == 1
        return operands[0]
    


class ControlFlowOperation(Operation):
    def __init__(self, operation: str):
        self.operation = operation
    
    def __str__(self):
        return self.operation
    
    def sprint(self, argument = None):
        if argument:
            return f"{self.operation} {argument}"
        return self.operation
    
    def deduce_type(self, arg: Constant | Variable) -> None:
        return None
    
class If(ControlFlowOperation):
    def __init__(self):
        super().__init__("if")

    def deduce_type(self, arg: Constant | Variable):
        if not isinstance(t := typeof(arg), (PrimitiveType, Pointer, UnknownType)):
            raise TypeDeductionError(f"If statement control expression must have scalar type but found {t}")

class LoopOp(ControlFlowOperation):
    def __init__(self):
        super().__init__("loop")

    def deduce_type(self, arg: Constant | Variable):
        if not isinstance(t := typeof(arg), (PrimitiveType, Pointer, UnknownType)):
            raise TypeDeductionError(f"Loop control expression must have scalar type but found {t}")

class Return(ControlFlowOperation):
    def __init__(self):
        super().__init__("return")

    def deduce_type(self, arg: Constant | Variable | None = None):
        if arg is not None and isinstance(t := typeof(arg), Array):
            raise TypeDeductionError(f"Can't return an array type (found {t})")

class Break(ControlFlowOperation):
    def __init__(self):
        super().__init__("break")

    def deduce_type(self):
        pass

class Continue(ControlFlowOperation):
    def __init__(self):
        super().__init__("continue")

    def deduce_type(self):
        pass


##################
# End Operations #
##################

VarOperand = Constant | Variable | CType
SSAOperand = tUnion[Constant, "SSAInstruction", Parameter, GlobalVariable, CType]
FUNCTION_CALL_OP = "function_call"
TERNARY_OP = ConditionalOperation()
COPY_OP = Copy()
STORE_OP = Store()
CAST_OP = Cast()
SUBSCRIPT_OP = Subscript()
TUPLE_INITIALIZER_OP = "tuple_init"
SIZEOF_OP = SizeOf()
RETURN_OP = Return()
IF_OP = If()
LOOP_OP = LoopOp()
BREAK_OP = Break()
CONTINUE_OP = Continue()
LOGICAL_NOT_OP = LogicalNot()
AND_OP = LogicalAnd()
OR_OP = LogicalOr()

#
# Instructions
#
class Instruction(ABC):
    """An instruction a single unit of computation that cannot be further broken down (at the 
    source level, which we preserve here).

    :param op: a symbol identifying what unit of computation this instruction object represents.
    """
    def __init__(self, op: Operation):
        self.op = op

class VarInstruction(Instruction):
    """A VarInstruction's inputs are variables and constants. Its output is stored in a variable.

    :param op: a symbol identifying what unit of computation this instruction object represents.
    :param result: the variable storing the result of the computation. Can be None if the result is not stored anywhere.
    :param operands: the input arguments to this instruction.
    :param ast_node: the tree_sitter AST node from which this instruction was derived, if any.

    Note that VarInstruction is not designed to handle multi-operation expressions like d = a * b + c;. Intermediate results
    should be stored in temporary variables, as in t1 = a * b; d = t1 + c;.
    """
    def __init__(self, op: Operation, result: Optional[Variable], operands: list[VarOperand], ast_node: Optional[Node] = None):
        super().__init__(op)
        self.result = result
        self.operands = operands
        self.ast_node = ast_node

    def __str__(self):
        return ("" if self.result is None else f"{self.result} = ") + self.op.sprint(*self.operands)
    
    def __repr__(self):
        op_names = [repr(op) for op in self.operands]
        if self.result is None:
            return f"{self.op} " + " ".join(op_names)
        return f"{self.result} = {self.op} " + " ".join(op_names)

class SSAInstruction(Instruction):
    """An instruction in Single Static Assignment form.

    :param op: a symbol identifying what unit of computation this instruction object represents.
    :param operands: the input arguments to this instruction.
    :param out_repr: a symbol representing the result of this instruction for display purposes.
    :param var_instruction: the VarInstruction from which this intruction was derived, if any.
    """
    def __init__(self, op: Operation, operands: list[SSAOperand], out_repr: str | None = None, var_instruction: Optional[VarInstruction] = None):
        super().__init__(op)
        self.operands = operands
        self.out_repr = out_repr
        self.var_instruction = var_instruction
        self.ast_node = var_instruction.ast_node if var_instruction is not None else None
        # Where the result of executing this instruction would be stored in memory. Declared
        # variables are considered valid storage locations, while temporary variables are not.
        # (Temporary variables are treated like temporary values stored only in registers and
        # never written to memory).
        self.storage = var_instruction.result if var_instruction is not None and var_instruction.result is not None and var_instruction.result.is_stack_allocated else None

    def __str__(self):
        printable_operands = []
        for operand in self.operands:
            if isinstance(operand, SSAInstruction):
                printable_operands.append(str(id(operand)) if operand.out_repr is None else operand.out_repr)
            else:
                printable_operands.append(str(operand))

        return ("" if self.out_repr is None else f"{self.out_repr} = ") + self.op.sprint(*printable_operands)
    
    def __repr__(self):
        op_names = []
        for op in self.operands:
            if isinstance(op, SSAInstruction):
                op_names.append(str(id(op)) if op.out_repr is None else op.out_repr)
            else:
                op_names.append(repr(op))
        if self.out_repr is None:
            return f"{self.op} " + " ".join(op_names)
        else:
            return f"{self.out_repr} = {self.op} " + " ".join(op_names)
    
    def __hash__(self):
        return id(self)


InsT = TypeVar('InsT', bound=Instruction)

#
# Basic Blocks
#
class BasicBlock(Generic[InsT]):
    id_counter = 0

    def __init__(self, instructions: list[InsT], predecessors: list['BasicBlock[InsT]'], successors: list['BasicBlock[InsT]']):
        self.instructions = instructions
        self.predecessors = predecessors
        self.successors = successors # Convention: if there are multiple successors, true block is first, false is second
        self.id = BasicBlock.id_counter
        BasicBlock.id_counter += 1

    def add_successor(self, successor: 'BasicBlock[InsT]'):
        self.successors.append(successor)
        successor.predecessors.append(self)
    
    def __iter__(self) -> Iterator[InsT]:
        """Iterate over the instructions in basic block in order.
        """
        for instruction in self.instructions:
            yield instruction

    def __str__(self):
        predecessors = ", ".join([str(p.id) for p in self.predecessors])
        instructions = "\n".join([str(instruction) for instruction in self])
        successors = ", ".join([str(s.id) for s in self.successors])

        return f"predecessors: {predecessors}\n ID = {self.id}\n{instructions}\nsuccessors: {successors}"
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    def __hash__(self):
        return id(self)

#
# Function
#
class Function(Generic[InsT]):
    def __init__(self, name: str, basic_blocks: list[BasicBlock[InsT]], parameters: list[Parameter], return_type: CType, node: Node):
        """Initialize a Function object.

        Precondition: The first element of basic_blocks is the functions' entry block.
        """
        assert(len(basic_blocks) > 0)
        assert(len(basic_blocks[0].predecessors) == 0)
        self.name = name
        self.entry_block = basic_blocks[0]
        self.basic_blocks = basic_blocks
        self.parameters = parameters
        self.return_type = return_type
        self.node = node

    def __iter__(self) -> Iterator[BasicBlock[InsT]]:
        """Iterate over the functions' basic blocks in an arbitrary order except the first block is the entry block.
        """
        for block in self.basic_blocks:
            yield block

    def __repr__(self) -> str:
        declaration = "function " + self.name + "(" + ", ".join([str(p) for p in self.parameters]) + ")\n"
        block_representations = [str(b) for b in self.basic_blocks]
        return declaration + "\n\n".join(block_representations)
