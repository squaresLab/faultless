"""External type inference for underspecified C fragments.

The ordinary ``Operation.infer_type`` methods in :mod:`faultless.ir` implement
local C typing rules: once operand declarations are known, they can infer an
expression result.  This module solves the inverse problem that appears when a
single function is compiled without the declarations for some globals or
callees.  It collects constraints from the whole function, propagates them to a
fixed point, and writes the preferred inferred types back to the IR objects.

The solver intentionally distinguishes exact type identity from C assignment or
call compatibility.  For example, ``double d; d = x;`` records that ``x`` is
assignable to ``double``; it does not equate the two declared object types.
When an unknown type must be rendered, the resolution phase chooses the exact
receiver type as a preference, because that is the most useful declaration to
emit, but known declarations are never rewritten by that preference.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Literal

from .ir import *


Requirement = Literal["arithmetic", "integer", "scalar", "pointer"]
SetMode = Literal["exact", "preferred"]
DiagnosticSubjectKind = Literal["variable", "function"]

_KNOWN_VARIADIC_PREFIX_ARITIES: dict[str, int] = {
    "printf": 1,
    "fprintf": 2,
    "sprintf": 2,
    "snprintf": 3,
    "scanf": 1,
    "fscanf": 2,
    "sscanf": 2,
}

_KNOWN_FUNCTION_TYPES: dict[str, FunctionType] = {
    "malloc": FunctionType(Pointer(Void()), [(SIZE_T, None)]),
    "calloc": FunctionType(Pointer(Void()), [(SIZE_T, None), (SIZE_T, None)]),
    "realloc": FunctionType(Pointer(Void()), [(Pointer(Void()), None), (SIZE_T, None)]),
}


@dataclass(frozen=True)
class TypeInferenceSubject:
    """A variable or external function slot involved in a diagnostic."""

    kind: DiagnosticSubjectKind
    name: str
    role: str | None = None

    def __str__(self) -> str:
        label = f"{self.kind} {self.name}"
        if self.role is not None:
            label += f" {self.role}"
        return label


@dataclass(frozen=True)
class TypeInferenceDiagnostic:
    """A human-readable explanation of a type inference conflict or warning."""

    message: str
    instruction: str | None = None
    subjects: tuple[TypeInferenceSubject, ...] = ()

    def __str__(self) -> str:
        parts = [self.message]
        if self.subjects:
            parts.append("subjects: " + ", ".join(str(subject) for subject in self.subjects))
        if self.instruction is not None:
            parts.append(f"instruction: {self.instruction}")
        return " | ".join(parts)


class TypeInferenceError(SemanticError):
    """Raised in strict mode when the collected type evidence is inconsistent."""

    def __init__(self, diagnostic: TypeInferenceDiagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass
class TypeInferenceResult:
    """Summary of facts materialized by :func:`infer_external_types`."""

    function_types: dict[object, FunctionType] = field(default_factory=dict)
    variable_types: dict[Variable, CType] = field(default_factory=dict)
    diagnostics: list[TypeInferenceDiagnostic] = field(default_factory=list)

    def external_function_declarations(self) -> list[str]:
        """Render inferred declarations for named external functions."""

        declarations = []
        for key, ftype in self.function_types.items():
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "function":
                declarations.append(ftype.declaration(str(key[1])) + ";")
        return declarations


@dataclass(eq=False)
class _TypeVariable:
    """A type variable in the external inference constraint graph."""

    ident: int
    name: str
    owner: Variable | None = None

    def __hash__(self) -> int:
        return self.ident


@dataclass
class _TypeState:
    """Mutable state for a union-find representative.

    ``typ`` is the current best hard/preferred type.  ``fixed`` means that the
    type came from a known declaration or literal and cannot be changed by
    preferences.  Requirements are category constraints such as "must be an
    integer" or "must be pointer-like"; they are checked independently from the
    preferred concrete rendering.
    """

    typ: CType | None = None
    fixed: bool = False
    owners: set[Variable] = field(default_factory=set)
    subjects: set[TypeInferenceSubject] = field(default_factory=set)
    requirements: set[Requirement] = field(default_factory=set)
    preferences: list[CType] = field(default_factory=list)
    is_null_pointer_constant: bool = False


class _ExternalFunctionSignature:
    """Shared signature variables for all call sites of one unknown callee."""

    def __init__(self, solver: "_TypeInferenceSolver", key: object, display_name: str):
        self.solver = solver
        self.key = key
        self.display_name = display_name
        self.subject = TypeInferenceSubject("function", display_name)
        self.return_var = solver.new_type_variable(
            f"{display_name}.ret",
            subject=TypeInferenceSubject("function", display_name, "return"),
        )
        self.parameter_vars: list[_TypeVariable] = []
        self.argument_vars: list[list[_TypeVariable]] = []
        self.arities: set[int] = set()
        self.inconsistent_arity = False
        self.return_value_used = False
        self.return_value_unused = False

    def observe_arity(self, arity: int, instruction: VarInstruction) -> None:
        if self.arities and arity not in self.arities and not self.solver.infer_variadic_functions:
            self.inconsistent_arity = True
            self.solver.warn(
                f"Cannot infer a fixed prototype for {self.display_name}: "
                f"observed both {min(self.arities)} and {arity} arguments.",
                instruction,
                subjects=(self.subject,),
            )
        self.arities.add(arity)

        while len(self.parameter_vars) < arity:
            idx = len(self.parameter_vars)
            self.parameter_vars.append(
                self.solver.new_type_variable(
                    f"{self.display_name}.arg{idx}",
                    subject=TypeInferenceSubject("function", self.display_name, f"arg{idx}"),
                )
            )

    def observe_call(self, operands: list[VarOperand], instruction: VarInstruction) -> None:
        self.observe_arity(len(operands), instruction)
        self.argument_vars.append([self.solver.slot_for_operand(operand) for operand in operands])

    def note_return_use(self, used: bool) -> None:
        self.return_value_used |= used
        self.return_value_unused |= not used

    def is_variadic(self) -> bool:
        return self.solver.infer_variadic_functions and (
            self.display_name in _KNOWN_VARIADIC_PREFIX_ARITIES or len(self.arities) > 1
        )

    def fixed_prefix_arity(self) -> int:
        if not self.is_variadic():
            return next(iter(self.arities)) if self.arities else 0
        known_prefix = _KNOWN_VARIADIC_PREFIX_ARITIES.get(self.display_name)
        if known_prefix is not None:
            return min(known_prefix, min(self.arities)) if self.arities else known_prefix

        fixed_arity = 0
        for idx in range(min(self.arities)):
            if not self._compatible_argument_prefix(idx):
                break
            fixed_arity += 1
        return fixed_arity

    def _compatible_argument_prefix(self, idx: int) -> bool:
        observed = [arguments[idx] for arguments in self.argument_vars if idx < len(arguments)]
        if len({self.solver.find(argument) for argument in observed}) == 1:
            return True

        for left_idx, left in enumerate(observed):
            left_t = self.solver.current_type(left)
            for right in observed[left_idx + 1:]:
                right_t = self.solver.current_type(right)
                if left_t is None or right_t is None:
                    return False
                left_null = self.solver.is_null_pointer_constant(left)
                right_null = self.solver.is_null_pointer_constant(right)
                if not (
                    _assignment_compatible(left_t, right_t, right_null, allow_same_width_integer_pointers=False)
                    or _assignment_compatible(right_t, left_t, left_null, allow_same_width_integer_pointers=False)
                ):
                    return False
        return True

    def function_type(self) -> FunctionType | None:
        """Resolve the accumulated signature into a concrete ``FunctionType``."""

        if not self.arities:
            return None
        if self.inconsistent_arity and not self.solver.infer_variadic_functions:
            return None
        is_variadic = self.is_variadic()
        fixed_arity = self.fixed_prefix_arity()

        if self.return_value_unused and not self.return_value_used:
            # An unused return value is only a soft preference.  If another
            # constraint later forces a non-void return, the normal resolver
            # below will keep the non-void type.
            self.solver.add_preference(self.return_var, Void())

        return_t = self.solver.resolved_type(self.return_var, allow_void=True)
        if return_t is None:
            return_t = UnknownType()

        params: list[tuple[CType | FunctionType.VariadicParameter, str | None]] = []
        for idx in range(fixed_arity):
            param_t = self.solver.resolved_type(self.parameter_vars[idx])
            params.append((param_t if param_t is not None else UnknownType(), None))
        if is_variadic:
            # Mixed arity call sites are modeled as a fixed common prefix plus
            # a C variadic tail.  The optional behavior is useful for printf-
            # style APIs while remaining disabled by default for stricter
            # inconsistent-call diagnostics.
            params.append((FunctionType.VariadicParameter(), None))

        return FunctionType(return_t, params)


class _Constraint:
    """Base class for solver constraints."""

    instruction: VarInstruction | None = None

    def variables(self) -> tuple[_TypeVariable, ...]:
        return ()

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        raise NotImplementedError

    def reason(self) -> str:
        return self.__class__.__name__


class _Equal(_Constraint):
    """Hard equality, implemented by merging union-find representatives."""

    def __init__(self, left: _TypeVariable, right: _TypeVariable, instruction: VarInstruction | None = None):
        self.left = left
        self.right = right
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.left, self.right)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        return solver.equate(self.left, self.right, self.instruction)


class _Known(_Constraint):
    """Restrict a variable to a specific known type."""

    def __init__(self, var: _TypeVariable, typ: CType, instruction: VarInstruction | None = None):
        self.var = var
        self.typ = typ
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.var,)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        return solver.set_type(self.var, self.typ, mode="exact", instruction=self.instruction)


class _Require(_Constraint):
    """Require a variable to belong to a C type category."""

    def __init__(self, var: _TypeVariable, requirement: Requirement, instruction: VarInstruction | None = None):
        self.var = var
        self.requirement: Requirement = requirement
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.var,)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        return solver.require(self.var, self.requirement, self.instruction)


class _PreferredType(_Constraint):
    """Soft concrete type evidence used to break otherwise ambiguous choices."""

    def __init__(self, var: _TypeVariable, typ: CType, instruction: VarInstruction | None = None):
        self.var = var
        self.typ = typ
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.var,)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        return solver.set_type(self.var, self.typ, mode="preferred", instruction=self.instruction)


class _Assignable(_Constraint):
    """C assignment compatibility from ``src`` expression to ``dst`` object."""

    def __init__(
        self,
        src: _TypeVariable,
        dst: _TypeVariable,
        *,
        src_operand: VarOperand | None = None,
        instruction: VarInstruction | None = None,
    ):
        self.src = src
        self.dst = dst
        self.src_operand = src_operand
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.src, self.dst)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        changed = False
        src_t = solver.current_type(self.src)
        dst_t = solver.current_type(self.dst)
        src_is_null = _is_null_pointer_constant(self.src_operand) or solver.is_null_pointer_constant(self.src)

        if dst_t is not None and _is_pointer_like(dst_t) and solver.has_requirement(self.src, "arithmetic") and not src_is_null:
            solver.contradiction(
                f"Cannot assign a non-constant arithmetic value to pointer type {dst_t}.",
                self.instruction,
                variables=(self.src, self.dst),
            )
            return False

        if src_t is not None and dst_t is not None:
            if not _assignment_compatible(dst_t, src_t, src_is_null):
                if (
                    isinstance(self.src_operand, IntegerConstant)
                    and _same_width_pointer_integer_types(dst_t, src_t)
                ):
                    return changed
                if (
                    _is_pointer_like(src_t)
                    and _is_pointer_like(dst_t)
                    and not src_is_null
                    and not solver.state(self.src).fixed
                ):
                    changed |= solver.set_type(self.src, Pointer(Void()), mode="preferred", instruction=self.instruction)
                    if _assignment_compatible(dst_t, solver.current_type(self.src) or src_t, src_is_null):
                        return changed
                solver.contradiction(
                    f"Type {src_t} is not assignable to {dst_t}.",
                    self.instruction,
                    variables=(self.src, self.dst),
                )
                return False
            if _is_pointer_like(src_t) and _is_pointer_like(dst_t) and _contains_unknown_type(src_t):
                changed |= solver.set_type(self.src, dst_t, mode="preferred", instruction=self.instruction)
            return changed

        if src_t is not None and dst_t is None:
            # A literal zero by itself is intentionally weak evidence.  It may
            # be an integer assignment or a null pointer constant; other
            # pointer evidence should decide.
            if src_is_null:
                solver.add_preference(self.dst, src_t if isinstance(src_t, Pointer) else INTEGER)
            else:
                changed |= solver.set_type(
                    self.dst,
                    _decay_array_type(src_t),
                    mode="preferred",
                    instruction=self.instruction,
                )

        if dst_t is not None and src_t is None:
            if src_is_null and _is_pointer_like(dst_t):
                return changed
            changed |= solver.set_type(self.src, dst_t, mode="preferred", instruction=self.instruction)

        if src_t is None and dst_t is None and solver.has_requirement(self.dst, "pointer") and not src_is_null:
            # Assignment to a known-pointer destination is real pointer
            # evidence for an unknown source expression.  This is what lets
            # ``q = p + 1; y = *q`` propagate the dereference of ``q`` back to
            # the result of ``p + 1`` and then to ``p``.
            changed |= solver.require(self.src, "pointer", self.instruction)

        return changed


class _CallCompatible(_Assignable):
    """Function-argument compatibility.

    Inferred prototypes use this mostly like assignment into the formal
    parameter, but repeated calls with incompatible object pointer arguments
    are generalized to ``void *``.  Known prototypes still use normal assignment
    compatibility because their parameter slots are fixed.
    """

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        src_t = solver.current_type(self.src)
        dst_t = solver.current_type(self.dst)
        src_is_null = _is_null_pointer_constant(self.src_operand) or solver.is_null_pointer_constant(self.src)

        if (
            src_t is not None
            and dst_t is not None
            and _is_pointer_like(src_t)
            and _is_pointer_like(dst_t)
            and _contains_unknown_type(dst_t)
            and not _contains_unknown_type(src_t)
            and not src_is_null
            and not solver.state(self.dst).fixed
        ):
            if solver.set_type(self.dst, _decay_array_type(src_t), mode="preferred", instruction=self.instruction):
                return True

        if src_t is not None and dst_t is not None and _same_width_pointer_integer_types(dst_t, src_t):
            return False

        if (
            src_t is not None
            and dst_t is not None
            and _is_pointer_like(src_t)
            and _is_pointer_like(dst_t)
            and not src_is_null
            and not _assignment_compatible(dst_t, src_t, src_is_null, allow_same_width_integer_pointers=False)
            and not solver.state(self.dst).fixed
        ):
            return solver.set_type(self.dst, Pointer(Void()), mode="preferred", instruction=self.instruction)

        return super().propagate(solver)


class _PointerTo(_Constraint):
    """Structural pointer relationship: ``ptr`` has pointee type ``pointee``."""

    def __init__(self, ptr: _TypeVariable, pointee: _TypeVariable, instruction: VarInstruction | None = None):
        self.ptr = ptr
        self.pointee = pointee
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.ptr, self.pointee)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        changed = solver.require(self.ptr, "pointer", self.instruction)
        ptr_t = solver.current_type(self.ptr)
        pointee_t = solver.current_type(self.pointee)

        if isinstance(ptr_t, Pointer):
            if isinstance(ptr_t.target_type, Void):
                # A void pointer tells us the pointer value can refer to some
                # object, but it does not make the addressed object itself
                # have type void. Keep a concrete object fallback available
                # for address-of/dereference idioms with otherwise unknown
                # globals.
                solver.add_preference(self.pointee, INTEGER)
            else:
                changed |= solver.set_type(self.pointee, ptr_t.target_type, mode="preferred", instruction=self.instruction)
        elif isinstance(ptr_t, Array):
            changed |= solver.set_type(self.pointee, ptr_t.element_type, mode="preferred", instruction=self.instruction)
        elif ptr_t is not None:
            solver.contradiction(
                f"Expected a pointer or array type but found {ptr_t}.",
                self.instruction,
                variables=(self.ptr, self.pointee),
            )

        if pointee_t is not None:
            # A known array base already satisfies pointer-to-style reads via
            # array-to-pointer decay.  Do not rewrite it to an exact pointer
            # type just because the pointee/result is known.
            if not isinstance(ptr_t, Array):
                changed |= solver.set_type(self.ptr, Pointer(pointee_t), mode="preferred", instruction=self.instruction)
        else:
            # A dereferenced but otherwise unconstrained object is ambiguous.
            # Prefer int as the same conservative default C uses for many
            # integer contexts, while retaining the pointer relationship.
            solver.add_preference(self.pointee, INTEGER)

        return changed


class _MemberAccessConstraint(_Constraint):
    """Structural aggregate relationship introduced by ``.`` and ``->``."""

    def __init__(
        self,
        base: _TypeVariable,
        field: Field,
        result: _TypeVariable,
        *,
        indirect: bool,
        instruction: VarInstruction | None = None,
    ):
        self.base = base
        self.field = field
        self.result = result
        self.indirect = indirect
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.base, self.result)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        changed = False
        result_t = solver.current_type(self.result)
        synthetic_field_t = result_t if result_t is not None else UnknownType()
        synthetic = Struct(None, [UDT.Field(synthetic_field_t, self.field.value)], defer_layout=True)

        if self.indirect:
            changed |= solver.require(self.base, "pointer", self.instruction)
            base_t = solver.current_type(self.base)
            if isinstance(base_t, Pointer):
                aggregate_t = _complete_udt_definition(base_t.target_type)
                if isinstance(aggregate_t, UnknownType) or (isinstance(aggregate_t, Struct) and aggregate_t.name is None):
                    changed |= solver.set_type(self.base, Pointer(synthetic), mode="exact", instruction=self.instruction)
                elif isinstance(aggregate_t, IncompleteStruct):
                    named_synthetic = Struct(aggregate_t.name, [UDT.Field(synthetic_field_t, self.field.value)], defer_layout=True)
                    changed |= solver.set_type(self.base, Pointer(named_synthetic), mode="exact", instruction=self.instruction)
                    aggregate_t = named_synthetic
                elif isinstance(aggregate_t, IncompleteUnion):
                    named_synthetic = Union(aggregate_t.name, [UDT.Field(synthetic_field_t, self.field.value)])
                    changed |= solver.set_type(self.base, Pointer(named_synthetic), mode="exact", instruction=self.instruction)
                    aggregate_t = named_synthetic
            elif base_t is None:
                aggregate_t = None
                changed |= solver.set_type(self.base, Pointer(synthetic), mode="exact", instruction=self.instruction)
            else:
                solver.contradiction(
                    f"Cannot apply indirect member access operator -> to non-pointer type {base_t}.",
                    self.instruction,
                    variables=(self.base,),
                )
                return changed
        else:
            aggregate_t = _complete_udt_definition(solver.current_type(self.base))
            if aggregate_t is None:
                changed |= solver.set_type(self.base, synthetic, mode="exact", instruction=self.instruction)
            elif isinstance(aggregate_t, Struct) and aggregate_t.name is None:
                changed |= solver.set_type(self.base, synthetic, mode="exact", instruction=self.instruction)
            elif isinstance(aggregate_t, IncompleteStruct):
                named_synthetic = Struct(aggregate_t.name, [UDT.Field(synthetic_field_t, self.field.value)], defer_layout=True)
                changed |= solver.set_type(self.base, named_synthetic, mode="exact", instruction=self.instruction)
                aggregate_t = named_synthetic
            elif isinstance(aggregate_t, IncompleteUnion):
                named_synthetic = Union(aggregate_t.name, [UDT.Field(synthetic_field_t, self.field.value)])
                changed |= solver.set_type(self.base, named_synthetic, mode="exact", instruction=self.instruction)
                aggregate_t = named_synthetic

        aggregate_t = _complete_udt_definition(solver.current_type(self.base))
        if self.indirect and isinstance(aggregate_t, Pointer):
            aggregate_t = _complete_udt_definition(aggregate_t.target_type)

        if isinstance(aggregate_t, (Struct, Union)):
            field_t = aggregate_t.typeof(self.field.value)
            if field_t is not None:
                result_t = solver.current_type(self.result)
                mode: SetMode = "exact"
                if isinstance(field_t, Array) and isinstance(result_t, Pointer) and not solver.state(self.result).fixed:
                    mode = "preferred"
                changed |= solver.set_type(self.result, field_t, mode=mode, instruction=self.instruction)
        elif aggregate_t is not None:
            solver.contradiction(
                f"Can only access member of struct or union but found {aggregate_t}.",
                self.instruction,
                variables=(self.base,),
            )

        return changed


class _UsualArithmetic(_Constraint):
    """C usual arithmetic conversions for binary arithmetic-like operators."""

    def __init__(
        self,
        lhs: _TypeVariable,
        rhs: _TypeVariable,
        result: _TypeVariable,
        *,
        require_integer: bool = False,
        instruction: VarInstruction | None = None,
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.result = result
        self.require_integer = require_integer
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.lhs, self.rhs, self.result)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        req: Requirement = "integer" if self.require_integer else "arithmetic"
        changed = solver.require(self.lhs, req, self.instruction)
        changed |= solver.require(self.rhs, req, self.instruction)
        changed |= solver.require(self.result, req, self.instruction)

        lhs_t = solver.current_type(self.lhs)
        rhs_t = solver.current_type(self.rhs)
        result_t = solver.current_type(self.result)

        if lhs_t is not None and not _is_arithmetic_type(lhs_t, self.require_integer):
            solver.contradiction(
                f"Expected arithmetic type but found {lhs_t}.",
                self.instruction,
                variables=(self.lhs,),
            )
            return changed
        if rhs_t is not None and not _is_arithmetic_type(rhs_t, self.require_integer):
            solver.contradiction(
                f"Expected arithmetic type but found {rhs_t}.",
                self.instruction,
                variables=(self.rhs,),
            )
            return changed

        if isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType):
            inferred = arithmetic_type_conversion(lhs_t, rhs_t)
            if self.require_integer and not isinstance(inferred, Integer):
                solver.contradiction(
                    f"Expected integer arithmetic but found {lhs_t} and {rhs_t}.",
                    self.instruction,
                    variables=(self.lhs, self.rhs),
                )
            result_mode: SetMode = "exact"
            if isinstance(result_t, PrimitiveType) and not solver.state(self.result).fixed:
                result_mode = "preferred"
            changed |= solver.set_type(self.result, inferred, mode=result_mode, instruction=self.instruction)
            return changed

        if isinstance(result_t, PrimitiveType):
            if isinstance(lhs_t, PrimitiveType) and rhs_t is None:
                changed |= solver.set_type(
                    self.rhs,
                    _preferred_other_arithmetic_operand(lhs_t, result_t, self.require_integer),
                    mode="preferred",
                    instruction=self.instruction,
                )
            elif isinstance(rhs_t, PrimitiveType) and lhs_t is None:
                changed |= solver.set_type(
                    self.lhs,
                    _preferred_other_arithmetic_operand(rhs_t, result_t, self.require_integer),
                    mode="preferred",
                    instruction=self.instruction,
                )
            elif lhs_t is None and rhs_t is None:
                changed |= solver.set_type(self.lhs, result_t, mode="preferred", instruction=self.instruction)
                changed |= solver.set_type(self.rhs, result_t, mode="preferred", instruction=self.instruction)

        return changed


class _Add(_Constraint):
    """Disjunctive C addition: arithmetic or pointer plus integer."""

    def __init__(self, lhs: _TypeVariable, rhs: _TypeVariable, result: _TypeVariable, instruction: VarInstruction | None = None):
        self.lhs = lhs
        self.rhs = rhs
        self.result = result
        self.instruction = instruction
        self.arithmetic = _UsualArithmetic(lhs, rhs, result, instruction=instruction)

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.lhs, self.rhs, self.result)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        lhs_t = solver.current_type(self.lhs)
        rhs_t = solver.current_type(self.rhs)
        result_t = solver.current_type(self.result)

        if _is_pointer_like(lhs_t) or solver.has_requirement(self.lhs, "pointer"):
            changed = solver.require(self.rhs, "integer", self.instruction)
            if lhs_t is not None:
                changed |= solver.set_type(self.result, _decay_array_type(lhs_t), mode="exact", instruction=self.instruction)
            else:
                changed |= solver.require(self.result, "pointer", self.instruction)
            return changed
        if _is_pointer_like(rhs_t) or solver.has_requirement(self.rhs, "pointer"):
            changed = solver.require(self.lhs, "integer", self.instruction)
            if rhs_t is not None:
                changed |= solver.set_type(self.result, _decay_array_type(rhs_t), mode="exact", instruction=self.instruction)
            else:
                changed |= solver.require(self.result, "pointer", self.instruction)
            return changed
        if _is_pointer_like(result_t) or solver.has_requirement(self.result, "pointer"):
            # In ``p + 1`` the integer side is normally syntactically obvious.
            # If both sides are unknown, prefer lhs as the pointer so that
            # common pointer-walk patterns infer stable declarations.
            rhs_is_integer = isinstance(rhs_t, Integer) or solver.has_requirement(self.rhs, "integer") or rhs_t is None
            lhs_is_integer = isinstance(lhs_t, Integer) or solver.has_requirement(self.lhs, "integer")
            if rhs_is_integer:
                changed = solver.require(self.lhs, "pointer", self.instruction)
                if result_t is not None:
                    changed |= solver.set_type(self.lhs, result_t, mode="preferred", instruction=self.instruction)
                changed |= solver.require(self.rhs, "integer", self.instruction)
                return changed
            if lhs_is_integer:
                changed = solver.require(self.rhs, "pointer", self.instruction)
                if result_t is not None:
                    changed |= solver.set_type(self.rhs, result_t, mode="preferred", instruction=self.instruction)
                changed |= solver.require(self.lhs, "integer", self.instruction)
                return changed

        if isinstance(result_t, PrimitiveType) or isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType):
            return self.arithmetic.propagate(solver)

        # With no hard evidence, addition remains ambiguous between arithmetic
        # addition and pointer arithmetic.  Seed preferences without committing
        # to either mode; later constraints will select the legal branch.
        solver.add_preference(self.lhs, INTEGER)
        solver.add_preference(self.rhs, INTEGER)
        solver.add_preference(self.result, INTEGER)
        return False


class _Subtract(_Constraint):
    """C subtraction, including pointer-minus-integer."""

    def __init__(self, lhs: _TypeVariable, rhs: _TypeVariable, result: _TypeVariable, instruction: VarInstruction | None = None):
        self.lhs = lhs
        self.rhs = rhs
        self.result = result
        self.instruction = instruction
        self.arithmetic = _UsualArithmetic(lhs, rhs, result, instruction=instruction)

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.lhs, self.rhs, self.result)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        lhs_t = solver.current_type(self.lhs)
        rhs_t = solver.current_type(self.rhs)
        result_t = solver.current_type(self.result)

        if (_is_pointer_like(lhs_t) or solver.has_requirement(self.lhs, "pointer")) and not _is_pointer_like(rhs_t):
            changed = solver.require(self.rhs, "integer", self.instruction)
            if lhs_t is not None:
                changed |= solver.set_type(self.result, _decay_array_type(lhs_t), mode="exact", instruction=self.instruction)
            else:
                changed |= solver.require(self.result, "pointer", self.instruction)
            return changed
        if _is_pointer_like(lhs_t) and _is_pointer_like(rhs_t):
            return solver.set_type(self.result, SIZE_T, mode="exact", instruction=self.instruction)
        if _is_pointer_like(result_t) or solver.has_requirement(self.result, "pointer"):
            changed = solver.require(self.lhs, "pointer", self.instruction)
            if result_t is not None:
                changed |= solver.set_type(self.lhs, result_t, mode="preferred", instruction=self.instruction)
            changed |= solver.require(self.rhs, "integer", self.instruction)
            return changed

        if isinstance(result_t, PrimitiveType) or isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType):
            return self.arithmetic.propagate(solver)

        solver.add_preference(self.lhs, INTEGER)
        solver.add_preference(self.rhs, INTEGER)
        solver.add_preference(self.result, INTEGER)
        return False


class _Shift(_Constraint):
    """C shift typing: both operands integer, result from promoted lhs."""

    def __init__(self, lhs: _TypeVariable, rhs: _TypeVariable, result: _TypeVariable, instruction: VarInstruction | None = None):
        self.lhs = lhs
        self.rhs = rhs
        self.result = result
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.lhs, self.rhs, self.result)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        changed = solver.require(self.lhs, "integer", self.instruction)
        changed |= solver.require(self.rhs, "integer", self.instruction)
        changed |= solver.require(self.result, "integer", self.instruction)

        lhs_t = solver.current_type(self.lhs)
        result_t = solver.current_type(self.result)
        if isinstance(lhs_t, Integer):
            result_mode: SetMode = "exact"
            if isinstance(result_t, PrimitiveType) and not solver.state(self.result).fixed:
                result_mode = "preferred"
            changed |= solver.set_type(self.result, integer_promotion(lhs_t), mode=result_mode, instruction=self.instruction)
        elif isinstance(result_t, Integer):
            changed |= solver.set_type(self.lhs, result_t, mode="preferred", instruction=self.instruction)
        return changed


class _Comparable(_Constraint):
    """Compatibility relation for relational and equality comparisons."""

    def __init__(
        self,
        lhs: _TypeVariable,
        rhs: _TypeVariable,
        *,
        lhs_operand: VarOperand | None = None,
        rhs_operand: VarOperand | None = None,
        equality: bool = False,
        instruction: VarInstruction | None = None,
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_operand = lhs_operand
        self.rhs_operand = rhs_operand
        self.equality = equality
        self.instruction = instruction

    def variables(self) -> tuple[_TypeVariable, ...]:
        return (self.lhs, self.rhs)

    def propagate(self, solver: "_TypeInferenceSolver") -> bool:
        lhs_t = solver.current_type(self.lhs)
        rhs_t = solver.current_type(self.rhs)

        lhs_null = _is_null_pointer_constant(self.lhs_operand)
        rhs_null = _is_null_pointer_constant(self.rhs_operand)
        changed = False

        if lhs_t is not None and rhs_t is not None:
            if not _comparison_compatible(lhs_t, rhs_t, lhs_null=lhs_null, rhs_null=rhs_null, equality=self.equality):
                solver.contradiction(
                    f"Types {lhs_t} and {rhs_t} are not comparable.",
                    self.instruction,
                    variables=(self.lhs, self.rhs),
                )
            return False

        if isinstance(lhs_t, (Pointer, Array)) and not rhs_null:
            changed |= solver.set_type(self.rhs, _decay_array_type(lhs_t), mode="preferred", instruction=self.instruction)
        elif isinstance(lhs_t, PrimitiveType) and rhs_t is None:
            changed |= solver.require(self.rhs, "arithmetic", self.instruction)
            if _is_weak_comparison_literal(self.lhs_operand):
                solver.add_preference(self.rhs, lhs_t)
            else:
                changed |= solver.set_type(self.rhs, lhs_t, mode="preferred", instruction=self.instruction)

        if isinstance(rhs_t, (Pointer, Array)) and not lhs_null:
            changed |= solver.set_type(self.lhs, _decay_array_type(rhs_t), mode="preferred", instruction=self.instruction)
        elif isinstance(rhs_t, PrimitiveType) and lhs_t is None:
            changed |= solver.require(self.lhs, "arithmetic", self.instruction)
            if _is_weak_comparison_literal(self.rhs_operand):
                solver.add_preference(self.lhs, rhs_t)
            else:
                changed |= solver.set_type(self.lhs, rhs_t, mode="preferred", instruction=self.instruction)

        if self.equality and (lhs_null or rhs_null):
            other = self.rhs if lhs_null else self.lhs
            solver.add_preference(other, Pointer(Void()))

        return changed


class _TypeInferenceSolver:
    """Constraint collector and worklist solver for one function."""

    def __init__(
        self,
        function: Function[VarInstruction],
        *,
        strict: bool = True,
        include_known_constraints: bool = False,
        infer_variadic_functions: bool = False,
    ):
        self.function = function
        self.strict = strict
        self.include_known_constraints = include_known_constraints
        self.infer_variadic_functions = infer_variadic_functions
        self._next_ident = 0
        self._var_slots: dict[Variable, _TypeVariable] = {}
        self._constant_slots: dict[int, _TypeVariable] = {}
        self._instruction_expr_slots: dict[VarInstruction, _TypeVariable] = {}
        self._parents: dict[_TypeVariable, _TypeVariable] = {}
        self._states: dict[_TypeVariable, _TypeState] = {}
        self.constraints: list[_Constraint] = []
        self.function_signatures: dict[object, _ExternalFunctionSignature] = {}
        self.use_counts = self._compute_use_counts()
        self.diagnostics: list[TypeInferenceDiagnostic] = []

    def new_type_variable(
        self,
        name: str,
        *,
        owner: Variable | None = None,
        subject: TypeInferenceSubject | None = None,
        initial_type: CType | None = None,
        fixed: bool = False,
    ) -> _TypeVariable:
        var = _TypeVariable(self._next_ident, name, owner)
        self._next_ident += 1
        self._parents[var] = var
        owners = {owner} if owner is not None else set()
        subjects = set()
        if owner is not None:
            subjects.add(TypeInferenceSubject("variable", owner.name))
        if subject is not None:
            subjects.add(subject)
        self._states[var] = _TypeState(initial_type, fixed, owners, subjects)
        return var

    def slot_for_operand(self, operand: VarOperand) -> _TypeVariable:
        if isinstance(operand, Variable):
            if operand not in self._var_slots:
                initial = None if isinstance(operand.type, UnknownType) else operand.type
                if operand.is_temporary:
                    producer = _producer_instruction(self.function, operand)
                    if producer is not None and isinstance(producer.op, FunctionCall):
                        initial = None
                decayed_temporary_array = operand.is_temporary and isinstance(initial, Array)
                if operand.is_temporary and initial is not None:
                    initial = _decay_array_type(initial)
                fixed = initial is not None and not _contains_unknown_type(initial) and not decayed_temporary_array
                self._var_slots[operand] = self.new_type_variable(
                    operand.name,
                    owner=operand,
                    initial_type=initial,
                    fixed=fixed,
                )
            return self._var_slots[operand]

        if isinstance(operand, CType):
            key = id(operand)
            if key not in self._constant_slots:
                self._constant_slots[key] = self.new_type_variable(str(operand), initial_type=operand, fixed=True)
            return self._constant_slots[key]

        key = id(operand)
        if key not in self._constant_slots:
            self._constant_slots[key] = self.new_type_variable(str(operand), initial_type=typeof(operand), fixed=True)
            if _is_null_pointer_constant(operand):
                self.mark_null_pointer_constant(self._constant_slots[key])
        return self._constant_slots[key]

    def expression_slot_for_instruction(self, instruction: VarInstruction) -> _TypeVariable:
        if instruction not in self._instruction_expr_slots:
            name = f"expr({instruction})"
            initial_type: CType | None = None
            if instruction.result is not None and instruction.result.is_temporary:
                # Temporary result variables represent expression values, not
                # storage locations.  If the ordinary local type deduction pass
                # has already typed that temporary, the external solver can
                # treat the expression slot as already known and avoid
                # rebuilding the same local constraint.
                initial_type = None if isinstance(instruction.result.type, UnknownType) else instruction.result.type
                if isinstance(instruction.op, FunctionCall):
                    initial_type = None
                decayed_temporary_array = isinstance(initial_type, Array)
                if initial_type is not None:
                    initial_type = _decay_array_type(initial_type)
            else:
                decayed_temporary_array = False
            self._instruction_expr_slots[instruction] = self.new_type_variable(
                name,
                initial_type=initial_type,
                fixed=initial_type is not None and _is_fully_known_type(initial_type) and not decayed_temporary_array,
            )
        return self._instruction_expr_slots[instruction]

    def find(self, var: _TypeVariable) -> _TypeVariable:
        parent = self._parents[var]
        if parent is not var:
            self._parents[var] = self.find(parent)
        return self._parents[var]

    def state(self, var: _TypeVariable) -> _TypeState:
        return self._states[self.find(var)]

    def current_type(self, var: _TypeVariable) -> CType | None:
        typ = self.state(var).typ
        if isinstance(typ, UnknownType):
            return None
        return typ

    def mark_null_pointer_constant(self, var: _TypeVariable) -> bool:
        state = self.state(var)
        if state.is_null_pointer_constant:
            return False
        state.is_null_pointer_constant = True
        return True

    def is_null_pointer_constant(self, var: _TypeVariable) -> bool:
        return self.state(var).is_null_pointer_constant

    def has_requirement(self, var: _TypeVariable, requirement: Requirement) -> bool:
        return requirement in self.state(var).requirements

    def diagnostic_subjects(
        self,
        variables: Iterable[_TypeVariable] = (),
        subjects: Iterable[TypeInferenceSubject] = (),
    ) -> tuple[TypeInferenceSubject, ...]:
        collected: dict[tuple[DiagnosticSubjectKind, str, str | None], TypeInferenceSubject] = {}
        for subject in subjects:
            collected[(subject.kind, subject.name, subject.role)] = subject
        for variable in variables:
            for subject in self.state(variable).subjects:
                collected[(subject.kind, subject.name, subject.role)] = subject
        return tuple(sorted(collected.values(), key=lambda subject: (subject.kind, subject.name, subject.role or "")))

    def warn(
        self,
        message: str,
        instruction: VarInstruction | None = None,
        *,
        variables: Iterable[_TypeVariable] = (),
        subjects: Iterable[TypeInferenceSubject] = (),
    ) -> None:
        self.diagnostics.append(
            TypeInferenceDiagnostic(
                message,
                _instruction_text(instruction),
                self.diagnostic_subjects(variables, subjects),
            )
        )

    def contradiction(
        self,
        message: str,
        instruction: VarInstruction | None = None,
        *,
        variables: Iterable[_TypeVariable] = (),
        subjects: Iterable[TypeInferenceSubject] = (),
    ) -> None:
        diagnostic = TypeInferenceDiagnostic(
            message,
            _instruction_text(instruction),
            self.diagnostic_subjects(variables, subjects),
        )
        if self.strict:
            raise TypeInferenceError(diagnostic)
        self.diagnostics.append(diagnostic)

    def add_constraint(self, constraint: _Constraint) -> None:
        if not self.include_known_constraints and self._constraint_is_fully_known(constraint):
            # Fully known constraints cannot improve external declarations.
            # Keeping them would make this pass duplicate local type checking
            # diagnostics that belong to the ordinary type deduction pass.
            return
        self.constraints.append(constraint)

    def _constraint_is_fully_known(self, constraint: _Constraint) -> bool:
        if isinstance(constraint, _MemberAccessConstraint):
            # A known aggregate can still be a partial synthetic aggregate from
            # an earlier member access. Later accesses may add fields.
            return False
        variables = constraint.variables()
        if isinstance(constraint, _Equal) and any(self.is_null_pointer_constant(var) for var in variables):
            # A casted null pointer constant can be fully typed, but equality
            # still carries the null-constant fact from the expression slot to
            # the temporary result slot.  That fact affects later pointer
            # assignment compatibility, so this is not redundant work.
            return False
        return bool(variables) and all(self._type_variable_is_fully_known(var) for var in variables)

    def _type_variable_is_fully_known(self, var: _TypeVariable) -> bool:
        typ = self.current_type(var)
        return typ is not None and _is_fully_known_type(typ)

    def equate(self, left: _TypeVariable, right: _TypeVariable, instruction: VarInstruction | None = None) -> bool:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot is rroot:
            return False

        lstate = self._states[lroot]
        rstate = self._states[rroot]
        merged_type = _exact_merge(lstate.typ, rstate.typ)
        if merged_type is None and lstate.typ is not None and rstate.typ is not None:
            self.contradiction(
                f"Cannot equate incompatible types {lstate.typ} and {rstate.typ}.",
                instruction,
                variables=(left, right),
            )
            return False

        # Keep the lower id as representative to make debugging deterministic.
        root, child = (lroot, rroot) if lroot.ident < rroot.ident else (rroot, lroot)
        root_state = self._states[root]
        child_state = self._states[child]
        root_state.typ = merged_type
        root_state.fixed = root_state.fixed or child_state.fixed
        root_state.owners |= child_state.owners
        root_state.subjects |= child_state.subjects
        root_state.preferences.extend(t for t in child_state.preferences if t not in root_state.preferences)
        root_state.requirements |= child_state.requirements
        root_state.is_null_pointer_constant = root_state.is_null_pointer_constant or child_state.is_null_pointer_constant
        self._parents[child] = root
        del self._states[child]
        self._check_requirements(root, instruction)
        return True

    def require(self, var: _TypeVariable, requirement: Requirement, instruction: VarInstruction | None = None) -> bool:
        state = self.state(var)
        requirements = _expanded_requirements(requirement)
        new_requirements = requirements - state.requirements
        if not new_requirements:
            return False

        if ("pointer" in requirements or "pointer" in state.requirements) and (
            "arithmetic" in requirements or "arithmetic" in state.requirements
        ):
            self.contradiction(
                "Type evidence requires both pointer and arithmetic categories.",
                instruction,
                variables=(var,),
            )
            return False

        state.requirements |= new_requirements

        # Requirements also seed ranking defaults so unconstrained externals
        # still render to useful declarations.
        if "integer" in new_requirements:
            self.add_preference(var, INTEGER)
        elif "arithmetic" in new_requirements:
            self.add_preference(var, INTEGER)
        elif "pointer" in new_requirements:
            self.add_preference(var, Pointer(UnknownType()))
            if state.typ is None:
                # Pointer-ness is a structural hard fact.  Recording a partial
                # pointer type immediately lets disjunctive constraints such as
                # addition choose the pointer-arithmetic branch on the next
                # worklist pass, while the pointee remains unresolved.
                state.typ = Pointer(UnknownType())

        self._check_requirements(var, instruction)
        return True

    def set_type(
        self,
        var: _TypeVariable,
        candidate: CType | None,
        *,
        mode: SetMode,
        instruction: VarInstruction | None = None,
    ) -> bool:
        if candidate is None or isinstance(candidate, UnknownType):
            return False

        state = self.state(var)
        if state.fixed and mode == "preferred":
            return False

        if not _type_satisfies_requirements(candidate, state.requirements):
            if mode == "exact":
                self.contradiction(
                    f"Type {candidate} does not satisfy required categories {sorted(state.requirements)}.",
                    instruction,
                    variables=(var,),
                )
            return False

        existing = state.typ
        if existing is None or isinstance(existing, UnknownType):
            state.typ = candidate
            return True
        if existing == candidate:
            return False

        merged = _exact_merge(existing, candidate) if mode == "exact" else _preferred_merge(existing, candidate)
        if merged is None:
            if mode == "exact":
                self.contradiction(
                    f"Conflicting type evidence: {existing} vs {candidate}.",
                    instruction,
                    variables=(var,),
                )
            return False
        if not _type_satisfies_requirements(merged, state.requirements):
            if mode == "exact":
                self.contradiction(
                    f"Type {merged} does not satisfy required categories {sorted(state.requirements)}.",
                    instruction,
                    variables=(var,),
                )
            return False

        if merged == existing:
            return False
        state.typ = merged
        return True

    def add_preference(self, var: _TypeVariable, typ: CType) -> None:
        state = self.state(var)
        if typ not in state.preferences:
            state.preferences.append(typ)

    def _check_requirements(self, var: _TypeVariable, instruction: VarInstruction | None = None) -> None:
        state = self.state(var)
        if "pointer" in state.requirements and "arithmetic" in state.requirements:
            self.contradiction(
                "Type evidence requires both pointer and arithmetic categories.",
                instruction,
                variables=(var,),
            )
        if state.typ is not None and not _type_satisfies_requirements(state.typ, state.requirements):
            self.contradiction(
                f"Type {state.typ} does not satisfy required categories {sorted(state.requirements)}.",
                instruction,
                variables=(var,),
            )

    def resolved_type(self, var: _TypeVariable, *, allow_void: bool = False) -> CType | None:
        state = self.state(var)
        typ = state.typ

        if typ is None or isinstance(typ, UnknownType):
            for preference in state.preferences:
                if not allow_void and isinstance(preference, Void):
                    continue
                if _type_satisfies_requirements(preference, state.requirements):
                    typ = _fill_unknowns(preference)
                    break

        if typ is None or isinstance(typ, UnknownType):
            if "pointer" in state.requirements:
                typ = Pointer(INTEGER)
            elif "integer" in state.requirements or "arithmetic" in state.requirements or "scalar" in state.requirements:
                typ = INTEGER
            else:
                # C's historical implicit-int default is not a good typing
                # rule for known source, but it is the most useful rendering
                # for a completely unconstrained external declaration.
                typ = INTEGER

        if typ is not None and _contains_unknown_type(typ):
            typ = _fill_unknowns(typ)
        if isinstance(typ, Void) and not allow_void:
            return None
        return typ

    def solve(self) -> None:
        """Run the constraint worklist to a fixed point."""

        worklist: deque[_Constraint] = deque(self.constraints)
        while worklist:
            constraint = worklist.popleft()
            if constraint.propagate(self):
                # The function IRs are small, and union-find can change a
                # constraint's representative.  Requeueing all constraints keeps
                # the implementation simple and deterministic.
                worklist.extend(self.constraints)

    def materialize(self) -> TypeInferenceResult:
        """Write resolved types back to IR variables and call operations."""

        result = TypeInferenceResult(diagnostics=list(self.diagnostics))

        # Resolve function signatures before variables.  Signature resolution
        # installs late preferences such as "unused return value defaults to
        # void"; temporary call-result variables share the same type variable
        # and should see those preferences before they are materialized.
        for signature in self.function_signatures.values():
            ftype = signature.function_type()
            if ftype is not None:
                result.function_types[signature.key] = ftype

        for var, slot in self._var_slots.items():
            resolved = self.resolved_type(slot, allow_void=var.is_temporary)
            if resolved is None:
                continue
            if (
                _contains_unknown_type(var.type)
                or isinstance(var.type, UnknownType)
                or (
                    var.is_temporary
                    and isinstance(var.type, Array)
                    and self._temporary_array_is_value_copy(var)
                )
                or (var.is_temporary and self._temporary_is_call_result(var))
            ):
                if var.type != resolved:
                    var.type = resolved
                    result.variable_types[var] = resolved

        for bb in self.function:
            for instruction in bb:
                if not isinstance(instruction.op, FunctionCall):
                    continue
                if instruction.op.ftype is not None:
                    continue
                key = _function_signature_key(instruction.op.fname)
                ftype = _known_function_type_for_call(instruction.op)
                if ftype is None:
                    ftype = result.function_types.get(key)
                if ftype is None and isinstance(instruction.op.fname, str):
                    ftype = _KNOWN_FUNCTION_TYPES.get(instruction.op.fname)
                if ftype is None:
                    continue
                instruction.op.ftype = ftype
                if (
                    isinstance(instruction.op.fname, Variable)
                    and (
                        isinstance(instruction.op.fname.type, UnknownType)
                        or _contains_unknown_type(instruction.op.fname.type)
                    )
                ):
                    instruction.op.fname.type = Pointer(ftype)
                    result.variable_types[instruction.op.fname] = Pointer(ftype)

        return result

    def _temporary_array_is_value_copy(self, var: Variable) -> bool:
        for bb in self.function:
            for instruction in bb:
                if instruction.result is var and isinstance(instruction.op, Copy):
                    return True
        return False

    def _temporary_is_call_result(self, var: Variable) -> bool:
        for bb in self.function:
            for instruction in bb:
                if instruction.result is var and isinstance(instruction.op, FunctionCall):
                    return True
        return False

    def _compute_use_counts(self) -> Counter[Variable]:
        counts: Counter[Variable] = Counter()
        for bb in self.function:
            for instruction in bb:
                for operand in instruction.operands:
                    if isinstance(operand, Variable):
                        counts[operand] += 1
        return counts


def _collect_constraints(solver: _TypeInferenceSolver) -> None:
    """Translate every IR instruction into solver constraints."""

    for bb in solver.function:
        for instruction in bb:
            _register_global_variables(solver, instruction)
            _register_function_call_signature(solver, instruction)

    for bb in solver.function:
        for instruction in bb:
            _collect_instruction_constraints(solver, instruction)


def _register_global_variables(solver: _TypeInferenceSolver, instruction: VarInstruction) -> None:
    """Ensure every mentioned global participates in final resolution.

    Some uses, notably ``sizeof(global)``, deliberately do not constrain the
    global's type.  The user-facing policy is still to render unconstrained
    external variables as ``int``, so globals need a solver slot even when no
    constraint references them.
    """

    for operand in instruction.operands:
        if isinstance(operand, GlobalVariable):
            solver.slot_for_operand(operand)
    if isinstance(instruction.result, GlobalVariable):
        solver.slot_for_operand(instruction.result)


def _register_function_call_signature(solver: _TypeInferenceSolver, instruction: VarInstruction) -> None:
    op = instruction.op
    if not isinstance(op, FunctionCall) or op.ftype is not None:
        return
    if _known_function_type_for_call(op) is not None:
        return
    if isinstance(op.fname, str) and op.fname in _KNOWN_FUNCTION_TYPES:
        return

    key = _function_signature_key(op.fname)
    signature = solver.function_signatures.get(key)
    if signature is None:
        signature = _ExternalFunctionSignature(solver, key, _function_display_name(op.fname))
        solver.function_signatures[key] = signature
    signature.observe_call(instruction.operands, instruction)


def _collect_instruction_constraints(solver: _TypeInferenceSolver, instruction: VarInstruction) -> None:
    op = instruction.op
    operands = instruction.operands
    result = instruction.result

    if isinstance(op, FunctionCall):
        expr = solver.expression_slot_for_instruction(instruction)
        _collect_function_call_constraints(solver, instruction, expr)
        _attach_result_storage(solver, instruction, expr)
        return

    if isinstance(op, Copy):
        if result is not None and len(operands) == 1:
            src = solver.slot_for_operand(operands[0])
            dst = solver.slot_for_operand(result)
            if result.is_temporary:
                if isinstance(typeof(operands[0]), Array):
                    solver.add_constraint(_Assignable(src, dst, src_operand=operands[0], instruction=instruction))
                else:
                    solver.add_constraint(_Equal(src, dst, instruction))
            else:
                solver.add_constraint(_Assignable(src, dst, src_operand=operands[0], instruction=instruction))
        return

    if isinstance(op, Store):
        if len(operands) == 2:
            lval = solver.slot_for_operand(operands[0])
            rval = solver.slot_for_operand(operands[1])
            solver.add_constraint(_Assignable(rval, lval, src_operand=operands[1], instruction=instruction))
            if result is not None:
                solver.add_constraint(_Equal(solver.slot_for_operand(result), lval, instruction))
        return

    if isinstance(op, Return):
        if len(operands) == 1:
            src = solver.slot_for_operand(operands[0])
            ret = solver.new_type_variable(f"{solver.function.name}.ret", initial_type=solver.function.return_type, fixed=not _contains_unknown_type(solver.function.return_type))
            solver.add_constraint(_Assignable(src, ret, src_operand=operands[0], instruction=instruction))
        elif isinstance(solver.function.return_type, UnknownType):
            ret = solver.new_type_variable(f"{solver.function.name}.ret")
            solver.add_preference(ret, Void())
        return

    if isinstance(op, (If, LoopOp)):
        if len(operands) == 1:
            solver.add_constraint(_Require(solver.slot_for_operand(operands[0]), "scalar", instruction))
        return

    if result is None:
        return

    expr = solver.expression_slot_for_instruction(instruction)

    if isinstance(op, Addition):
        solver.add_constraint(_Add(solver.slot_for_operand(operands[0]), solver.slot_for_operand(operands[1]), expr, instruction))
    elif isinstance(op, Subtraction):
        solver.add_constraint(_Subtract(solver.slot_for_operand(operands[0]), solver.slot_for_operand(operands[1]), expr, instruction))
    elif isinstance(op, MultiplicativeOperator):
        solver.add_constraint(
            _UsualArithmetic(
                solver.slot_for_operand(operands[0]),
                solver.slot_for_operand(operands[1]),
                expr,
                require_integer=isinstance(op, ModulusDivision),
                instruction=instruction,
            )
        )
    elif isinstance(op, Bitwise):
        solver.add_constraint(
            _UsualArithmetic(
                solver.slot_for_operand(operands[0]),
                solver.slot_for_operand(operands[1]),
                expr,
                require_integer=True,
                instruction=instruction,
            )
        )
    elif isinstance(op, BitShift):
        solver.add_constraint(_Shift(solver.slot_for_operand(operands[0]), solver.slot_for_operand(operands[1]), expr, instruction))
    elif isinstance(op, RelationalOperation):
        solver.add_constraint(_Known(expr, INTEGER, instruction))
        solver.add_constraint(
            _Comparable(
                solver.slot_for_operand(operands[0]),
                solver.slot_for_operand(operands[1]),
                lhs_operand=operands[0],
                rhs_operand=operands[1],
                instruction=instruction,
            )
        )
    elif isinstance(op, EqualityOperation):
        solver.add_constraint(_Known(expr, INTEGER, instruction))
        solver.add_constraint(
            _Comparable(
                solver.slot_for_operand(operands[0]),
                solver.slot_for_operand(operands[1]),
                lhs_operand=operands[0],
                rhs_operand=operands[1],
                equality=True,
                instruction=instruction,
            )
        )
    elif isinstance(op, LogicalOperator):
        solver.add_constraint(_Known(expr, INTEGER, instruction))
        solver.add_constraint(_Require(solver.slot_for_operand(operands[0]), "scalar", instruction))
        solver.add_constraint(_Require(solver.slot_for_operand(operands[1]), "scalar", instruction))
    elif isinstance(op, LogicalNot):
        solver.add_constraint(_Known(expr, INTEGER, instruction))
        solver.add_constraint(_Require(solver.slot_for_operand(operands[0]), "scalar", instruction))
    elif isinstance(op, UnaryMinus):
        solver.add_constraint(_Require(solver.slot_for_operand(operands[0]), "arithmetic", instruction))
        solver.add_constraint(_Require(expr, "arithmetic", instruction))
        solver.add_constraint(_Assignable(solver.slot_for_operand(operands[0]), expr, src_operand=operands[0], instruction=instruction))
    elif isinstance(op, BitwiseNot):
        solver.add_constraint(_Require(solver.slot_for_operand(operands[0]), "integer", instruction))
        solver.add_constraint(_Require(expr, "integer", instruction))
        solver.add_constraint(_Assignable(solver.slot_for_operand(operands[0]), expr, src_operand=operands[0], instruction=instruction))
    elif isinstance(op, AddressOf):
        solver.add_constraint(_PointerTo(expr, solver.slot_for_operand(operands[0]), instruction))
    elif isinstance(op, Dereference):
        solver.add_constraint(_PointerTo(solver.slot_for_operand(operands[0]), expr, instruction))
    elif isinstance(op, Subscript):
        solver.add_constraint(_Require(solver.slot_for_operand(operands[1]), "integer", instruction))
        solver.add_constraint(_PointerTo(solver.slot_for_operand(operands[0]), expr, instruction))
    elif isinstance(op, MemberAccess):
        _collect_member_access_constraints(solver, instruction, expr)
    elif isinstance(op, Cast):
        target = operands[0]
        if isinstance(target, CType):
            solver.add_constraint(_Known(expr, target, instruction))
            if len(operands) > 1:
                value = solver.slot_for_operand(operands[1])
                solver.add_constraint(_Require(value, "scalar", instruction))
                if isinstance(target, Pointer) and (_is_null_pointer_constant(operands[1]) or solver.is_null_pointer_constant(value)):
                    solver.mark_null_pointer_constant(expr)
                elif isinstance(target, Pointer):
                    _collect_pointer_cast_byte_base_constraints(solver, instruction, operands[1], value)
    elif isinstance(op, SizeOf):
        solver.add_constraint(_Known(expr, SIZE_T, instruction))
    elif isinstance(op, Initializer):
        solver.add_constraint(_Known(expr, op.type, instruction))
        _collect_initializer_constraints(solver, op, operands, instruction)
    elif isinstance(op, Phi):
        for operand in operands:
            solver.add_constraint(_Assignable(solver.slot_for_operand(operand), expr, src_operand=operand, instruction=instruction))
            solver.add_constraint(_Assignable(expr, solver.slot_for_operand(operand), instruction=instruction))

    _attach_result_storage(solver, instruction, expr)


def _attach_result_storage(solver: _TypeInferenceSolver, instruction: VarInstruction, expr: _TypeVariable) -> None:
    """Connect an expression result to the IR result variable.

    Temporary result variables are expression SSA values, so they are equal to
    the expression slot.  Named locals and globals are storage locations; the
    expression only needs to be assignable to that object.
    """

    result = instruction.result
    if result is None:
        return
    result_slot = solver.slot_for_operand(result)
    if result.is_temporary:
        solver.add_constraint(_Equal(expr, result_slot, instruction))
    else:
        solver.add_constraint(_Assignable(expr, result_slot, instruction=instruction))


def _collect_function_call_constraints(
    solver: _TypeInferenceSolver,
    instruction: VarInstruction,
    expr: _TypeVariable,
) -> None:
    op = instruction.op
    assert isinstance(op, FunctionCall)

    ftype = op.ftype
    if ftype is None:
        ftype = _known_function_type_for_call(op)
    if ftype is None and isinstance(op.fname, str):
        ftype = _KNOWN_FUNCTION_TYPES.get(op.fname)

    if ftype is not None:
        solver.add_constraint(_Known(expr, ftype.return_type, instruction))
        for operand, (param_t, _) in zip(instruction.operands, ftype.parameters):
            if isinstance(param_t, FunctionType.VariadicParameter):
                break
            param_slot = solver.new_type_variable(str(param_t), initial_type=_decay_array_type(param_t), fixed=True)
            solver.add_constraint(
                _CallCompatible(solver.slot_for_operand(operand), param_slot, src_operand=operand, instruction=instruction)
            )
        return

    key = _function_signature_key(op.fname)
    signature = solver.function_signatures[key]
    signature.note_return_use(_call_return_is_used(solver, instruction))
    solver.add_constraint(_Equal(expr, signature.return_var, instruction))

    fixed_arity = signature.fixed_prefix_arity() if signature.is_variadic() else len(instruction.operands)
    for idx, operand in enumerate(instruction.operands):
        if idx >= fixed_arity:
            break
        arg_slot = solver.slot_for_operand(operand)
        param_slot = signature.parameter_vars[idx]
        solver.add_constraint(_CallCompatible(arg_slot, param_slot, src_operand=operand, instruction=instruction))


def _collect_pointer_cast_byte_base_constraints(
    solver: _TypeInferenceSolver,
    instruction: VarInstruction,
    value_operand: VarOperand,
    value_slot: _TypeVariable,
) -> None:
    """Recognize ``(T *)(base + byte_offset)`` as address-base evidence.

    Hex-Rays often emits byte-address arithmetic against an untyped global,
    then casts the result to the final pointer type before dereferencing.  The
    expression before the cast is not C pointer arithmetic, so the base should
    be modeled as ``void *`` rather than ``T *``; otherwise Faultless would
    scale the explicit byte offset a second time.
    """

    if not isinstance(value_operand, Variable):
        return

    producer = _producer_instruction(solver.function, value_operand)
    if producer is None or not isinstance(producer.op, (Addition, Subtraction)) or len(producer.operands) != 2:
        return

    left, right = producer.operands
    if isinstance(producer.op, Subtraction):
        candidates = ((left, right),)
    else:
        candidates = ((left, right), (right, left))

    for base_operand, offset_operand in candidates:
        if not isinstance(base_operand, Variable) or not _is_unknown_external_base(solver.function, base_operand):
            continue
        if isinstance(offset_operand, CType):
            continue
        offset_t = typeof(offset_operand)
        if not isinstance(offset_t, (Integer, UnknownType)):
            continue

        byte_pointer = Pointer(Void())
        base = solver.slot_for_operand(base_operand)
        solver.add_constraint(_Require(value_slot, "pointer", instruction))
        solver.add_constraint(_PreferredType(value_slot, byte_pointer, instruction))
        solver.add_constraint(_Require(base, "pointer", producer))
        solver.add_constraint(_PreferredType(base, byte_pointer, producer))
        return


def _producer_instruction(function: Function[VarInstruction], variable: Variable) -> VarInstruction | None:
    for bb in function:
        for instruction in bb:
            if instruction.result is variable:
                return instruction
    return None


def _is_unknown_external_base(function: Function[VarInstruction], operand: Variable) -> bool:
    if not isinstance(operand, GlobalVariable) or not isinstance(operand.type, UnknownType):
        return False

    for bb in function:
        for instruction in bb:
            for used_operand in instruction.operands:
                if used_operand is not operand:
                    continue
                if (
                    isinstance(instruction.op, (Addition, Subtraction))
                    and isinstance(instruction.result, Variable)
                    and _is_pointer_cast_operand(function, instruction.result)
                ):
                    continue
                return False
    return True


def _is_pointer_cast_operand(function: Function[VarInstruction], operand: Variable) -> bool:
    for bb in function:
        for instruction in bb:
            if (
                isinstance(instruction.op, Cast)
                and len(instruction.operands) > 1
                and isinstance(instruction.operands[0], Pointer)
                and instruction.operands[1] is operand
            ):
                return True
    return False


def _collect_member_access_constraints(
    solver: _TypeInferenceSolver,
    instruction: VarInstruction,
    expr: _TypeVariable,
) -> None:
    op = instruction.op
    assert isinstance(op, MemberAccess)
    if len(instruction.operands) != 2 or not isinstance(instruction.operands[1], Field):
        return

    base = solver.slot_for_operand(instruction.operands[0])
    field = instruction.operands[1]
    base_t = solver.current_type(base)

    solver.add_constraint(
        _MemberAccessConstraint(
            base,
            field,
            expr,
            indirect=op.indirect,
            instruction=instruction,
        )
    )

    if op.indirect:
        solver.add_constraint(_Require(base, "pointer", instruction))
        if isinstance(base_t, Pointer) and isinstance(base_t.target_type, (Struct, Union)):
            field_t = base_t.target_type.typeof(field.value)
            if field_t is not None:
                solver.add_constraint(_Known(expr, field_t, instruction))
    else:
        if isinstance(base_t, (Struct, Union)):
            field_t = base_t.typeof(field.value)
            if field_t is not None:
                solver.add_constraint(_Known(expr, field_t, instruction))


def _collect_initializer_constraints(
    solver: _TypeInferenceSolver,
    op: Initializer,
    operands: list[VarOperand],
    instruction: VarInstruction,
) -> None:
    if isinstance(op.type, Array):
        element_slot = solver.new_type_variable(str(op.type.element_type), initial_type=op.type.element_type, fixed=True)
        for operand in operands:
            solver.add_constraint(_Assignable(solver.slot_for_operand(operand), element_slot, src_operand=operand, instruction=instruction))
    elif isinstance(op.type, (Struct, Union)) and op.field_names is not None:
        for field_name, operand in zip(op.field_names, operands):
            field_t = op.type.typeof(field_name)
            if field_t is not None:
                field_slot = solver.new_type_variable(str(field_t), initial_type=field_t, fixed=True)
                solver.add_constraint(_Assignable(solver.slot_for_operand(operand), field_slot, src_operand=operand, instruction=instruction))
    elif isinstance(op.type, (PrimitiveType, Pointer)) and len(operands) == 1:
        dst = solver.new_type_variable(str(op.type), initial_type=op.type, fixed=True)
        solver.add_constraint(_Assignable(solver.slot_for_operand(operands[0]), dst, src_operand=operands[0], instruction=instruction))


def _call_return_is_used(solver: _TypeInferenceSolver, instruction: VarInstruction) -> bool:
    result = instruction.result
    if result is None:
        return False
    if not result.is_temporary:
        return True
    return solver.use_counts[result] > 0


def _function_signature_key(fname: str | Variable | SSAInstruction) -> object:
    if isinstance(fname, str):
        return ("function", fname)
    return ("function-pointer", fname)


def _known_function_type_for_call(op: FunctionCall) -> FunctionType | None:
    fname = op.fname
    if isinstance(fname, Variable) and isinstance(fname.type, Pointer) and isinstance(fname.type.target_type, FunctionType):
        return fname.type.target_type
    return None


def _function_display_name(fname: str | Variable | SSAInstruction) -> str:
    if isinstance(fname, str):
        return fname
    return str(fname)


def _instruction_text(instruction: VarInstruction | None) -> str | None:
    return None if instruction is None else str(instruction)


def _expanded_requirements(requirement: Requirement) -> set[Requirement]:
    if requirement == "integer":
        return {"integer", "arithmetic", "scalar"}
    if requirement == "arithmetic":
        return {"arithmetic", "scalar"}
    if requirement == "pointer":
        return {"pointer", "scalar"}
    return {"scalar"}


def _type_satisfies_requirements(typ: CType, requirements: set[Requirement]) -> bool:
    if not requirements:
        return True
    if "pointer" in requirements and not _is_pointer_like(typ):
        return False
    if "integer" in requirements and not isinstance(typ, Integer):
        return False
    if "arithmetic" in requirements and not isinstance(typ, PrimitiveType):
        return False
    if "scalar" in requirements and not isinstance(typ, (PrimitiveType, Pointer, Array)):
        return False
    return True


def _is_arithmetic_type(typ: CType, require_integer: bool) -> bool:
    return isinstance(typ, Integer) if require_integer else isinstance(typ, PrimitiveType)


def _is_pointer_like(typ: CType | None) -> bool:
    return isinstance(typ, (Pointer, Array))


def _is_null_pointer_constant(operand: VarOperand | None) -> bool:
    return isinstance(operand, IntegerConstant) and operand.value == 0


def _is_weak_comparison_literal(operand: VarOperand | None) -> bool:
    return isinstance(operand, (IntegerConstant, FloatConstant, CharLiteral))


def _decay_array_type(typ: CType) -> CType:
    return Pointer(typ.element_type) if isinstance(typ, Array) else typ


def _complete_udt_definition(typ: CType | None) -> CType | None:
    if isinstance(typ, (IncompleteStruct, IncompleteUnion)) and typ.full_definition is not None:
        return typ.full_definition
    return typ


def _assignment_compatible(
    dst_t: CType,
    src_t: CType,
    src_is_null: bool,
    *,
    allow_same_width_integer_pointers: bool = True,
) -> bool:
    if isinstance(dst_t, UnknownType) or isinstance(src_t, UnknownType):
        return True
    src_t = _decay_array_type(src_t)
    if dst_t == src_t:
        return True
    if isinstance(dst_t, PrimitiveType) and isinstance(src_t, PrimitiveType):
        return True
    if _same_width_decompiler_placeholder_assignment(dst_t, src_t):
        return True
    if compatible_pointer_types(dst_t, src_t):
        return True
    if allow_same_width_integer_pointers and _same_width_integer_pointer_types(dst_t, src_t):
        return True
    if isinstance(dst_t, Pointer) and src_is_null:
        return True
    return False


def _same_width_pointer_integer_types(left: CType, right: CType) -> bool:
    if isinstance(left, Pointer) and isinstance(right, Integer):
        return left.get_size() == right.get_size()
    if isinstance(right, Pointer) and isinstance(left, Integer):
        return right.get_size() == left.get_size()
    return False


def _same_width_decompiler_placeholder_assignment(dst_t: CType, src_t: CType) -> bool:
    return (
        is_decompiler_placeholder_type(dst_t)
        and isinstance(src_t, (PrimitiveType, Pointer))
        and dst_t.get_size() == src_t.get_size()
    ) or (
        is_decompiler_placeholder_type(src_t)
        and isinstance(dst_t, (PrimitiveType, Pointer))
        and src_t.get_size() == dst_t.get_size()
    )


def _same_width_integer_pointer_types(left: CType, right: CType) -> bool:
    if not isinstance(left, Pointer) or not isinstance(right, Pointer):
        return False
    left_target = _complete_udt_definition(left.target_type)
    right_target = _complete_udt_definition(right.target_type)
    return (
        isinstance(left_target, Integer)
        and isinstance(right_target, Integer)
        and left_target.size == right_target.size
    )


def _comparison_compatible(
    lhs_t: CType,
    rhs_t: CType,
    *,
    lhs_null: bool,
    rhs_null: bool,
    equality: bool,
) -> bool:
    lhs_t = _decay_array_type(lhs_t)
    rhs_t = _decay_array_type(rhs_t)
    if isinstance(lhs_t, PrimitiveType) and isinstance(rhs_t, PrimitiveType):
        return True
    if compatible_pointer_types(lhs_t, rhs_t, wild_void=equality):
        return True
    if equality and ((isinstance(lhs_t, Pointer) and rhs_null) or (isinstance(rhs_t, Pointer) and lhs_null)):
        return True
    return False


def _preferred_other_arithmetic_operand(known: PrimitiveType, result: PrimitiveType, require_integer: bool) -> PrimitiveType:
    if require_integer and not isinstance(result, Integer):
        return INTEGER
    if arithmetic_type_conversion(known, result) == result:
        return result
    return INTEGER if require_integer else result


def _exact_merge(existing: CType | None, candidate: CType | None) -> CType | None:
    if candidate is None or isinstance(candidate, UnknownType):
        return existing
    if existing is None or isinstance(existing, UnknownType):
        return candidate
    if existing == candidate:
        return existing

    if isinstance(existing, Pointer) and isinstance(candidate, Pointer):
        target = _exact_merge(existing.target_type, candidate.target_type)
        return Pointer(target) if target is not None else None
    if isinstance(existing, Array) and isinstance(candidate, Array) and existing.nelements == candidate.nelements:
        element = _exact_merge(existing.element_type, candidate.element_type)
        return Array(element, existing.nelements) if element is not None else None
    if isinstance(existing, Struct) and isinstance(candidate, Struct) and existing.name == candidate.name:
        return _merge_struct_types(existing, candidate)
    if isinstance(existing, FunctionType) and isinstance(candidate, FunctionType) and len(existing.parameters) == len(candidate.parameters):
        ret = _exact_merge(existing.return_type, candidate.return_type)
        if ret is None:
            return None
        params = []
        for (left_t, left_name), (right_t, _) in zip(existing.parameters, candidate.parameters):
            if isinstance(left_t, FunctionType.VariadicParameter) or isinstance(right_t, FunctionType.VariadicParameter):
                if left_t != right_t:
                    return None
                params.append((left_t, left_name))
                continue
            param_t = _exact_merge(left_t, right_t)
            if param_t is None:
                return None
            params.append((param_t, left_name))
        return FunctionType(ret, params)
    if _same_incomplete_and_complete_udt(existing, candidate):
        return candidate
    if _same_incomplete_and_complete_udt(candidate, existing):
        return existing
    return None


def _merge_struct_types(existing: Struct, candidate: Struct) -> Struct | None:
    fields: dict[str, UDT.Field] = {field.name: field for field in existing.members}
    order = [field.name for field in existing.members]
    changed = False

    for candidate_field in candidate.members:
        existing_field = fields.get(candidate_field.name)
        if existing_field is None:
            fields[candidate_field.name] = candidate_field
            order.append(candidate_field.name)
            changed = True
            continue

        merged_t = _exact_merge(existing_field.type, candidate_field.type)
        if merged_t is None:
            merged_t = _merge_synthetic_struct_field_types(existing_field.type, candidate_field.type)
        if merged_t is None:
            return None
        if merged_t != existing_field.type:
            fields[candidate_field.name] = UDT.Field(merged_t, candidate_field.name)
            changed = True

    if not changed:
        return existing
    return Struct(existing.name, [fields[name] for name in order], defer_layout=True)


def _merge_synthetic_struct_field_types(existing: CType, candidate: CType) -> CType | None:
    if isinstance(existing, Pointer) and isinstance(candidate, Pointer):
        target = _merge_synthetic_struct_field_types(existing.target_type, candidate.target_type)
        if target is not None:
            return Pointer(target)
        if compatible_pointer_types(existing, candidate, wild_void=True):
            if _contains_void_pointer_target(existing) and not _contains_void_pointer_target(candidate):
                return candidate
            return existing
    if isinstance(existing, Void):
        return candidate
    if isinstance(candidate, Void):
        return existing
    return None


def _contains_void_pointer_target(typ: CType) -> bool:
    return isinstance(typ, Pointer) and (
        isinstance(typ.target_type, Void) or _contains_void_pointer_target(typ.target_type)
    )


def _preferred_anonymous_struct_merge(existing: CType, candidate: CType) -> Struct | None:
    if not (
        isinstance(existing, Struct)
        and isinstance(candidate, Struct)
        and existing.name is None
        and candidate.name is None
    ):
        return None

    existing_fields = {field.name: field for field in existing.members}
    candidate_fields = {field.name: field for field in candidate.members}
    if not set(existing_fields).issubset(candidate_fields):
        return None

    merged_fields: list[UDT.Field] = []
    for candidate_field in candidate.members:
        existing_field = existing_fields.get(candidate_field.name)
        if existing_field is None:
            merged_fields.append(candidate_field)
            continue

        merged_t = _exact_merge(existing_field.type, candidate_field.type)
        if merged_t is None:
            merged_t = _merge_synthetic_struct_field_types(existing_field.type, candidate_field.type)
        if merged_t is None:
            return None
        merged_fields.append(UDT.Field(merged_t, candidate_field.name))

    return Struct(None, merged_fields, defer_layout=True)


def _preferred_merge(existing: CType, candidate: CType) -> CType | None:
    preferred_struct = _preferred_anonymous_struct_merge(existing, candidate)
    if preferred_struct is not None:
        return preferred_struct
    exact = _exact_merge(existing, candidate)
    if exact is not None:
        return exact
    if isinstance(existing, Array) and isinstance(candidate, Pointer):
        existing = Pointer(existing.element_type)
    if isinstance(existing, Pointer) and isinstance(candidate, Array):
        candidate = Pointer(candidate.element_type)
    if isinstance(existing, Pointer) and isinstance(candidate, Pointer):
        return Pointer(_preferred_pointer_target(existing.target_type, candidate.target_type))
    if isinstance(existing, PrimitiveType) and isinstance(candidate, PrimitiveType):
        return arithmetic_type_conversion(existing, candidate)
    return existing


def _preferred_pointer_target(existing: CType, candidate: CType) -> CType:
    preferred_struct = _preferred_anonymous_struct_merge(existing, candidate)
    if preferred_struct is not None:
        return preferred_struct
    exact = _exact_merge(existing, candidate)
    if exact is not None:
        return exact
    if isinstance(existing, Pointer) and isinstance(candidate, Pointer):
        return Pointer(_preferred_pointer_target(existing.target_type, candidate.target_type))
    if isinstance(existing, PrimitiveType) and isinstance(candidate, PrimitiveType):
        return arithmetic_type_conversion(existing, candidate)
    if isinstance(existing, Void) or isinstance(candidate, Void):
        return Void()
    return Void()


def _same_incomplete_and_complete_udt(left: CType, right: CType) -> bool:
    return isinstance(left, IncompleteUDT) and isinstance(right, UDT) and left.name == right.name


def _is_fully_known_type(typ: CType | FunctionType.VariadicParameter) -> bool:
    """Return True when a C type has no unresolved pieces.

    The external solver represents unknown declarations with type variables,
    but partially solved structural types can still contain ``UnknownType``
    recursively, for example ``int *`` is fully known while
    ``UnknownType *`` is not.
    """

    return not _contains_unknown_type(typ)


def _contains_unknown_type(typ: CType | FunctionType.VariadicParameter) -> bool:
    if isinstance(typ, UnknownType):
        return True
    if isinstance(typ, Pointer):
        return _contains_unknown_type(typ.target_type)
    if isinstance(typ, Array):
        return _contains_unknown_type(typ.element_type)
    if isinstance(typ, Struct):
        return any(_contains_unknown_type(field.type) for field in typ.members)
    if isinstance(typ, FunctionType):
        return _contains_unknown_type(typ.return_type) or any(
            _contains_unknown_type(param_t) for param_t, _ in typ.parameters
        )
    return False


def _fill_unknowns(typ: CType) -> CType:
    """Replace unresolved structural unknowns with stable default renderings."""

    if isinstance(typ, UnknownType):
        return INTEGER
    if isinstance(typ, Pointer):
        return Pointer(_fill_unknowns(typ.target_type))
    if isinstance(typ, Array):
        return Array(_fill_unknowns(typ.element_type), typ.nelements)
    if isinstance(typ, Struct):
        return Struct(
            typ.name,
            [UDT.Field(_fill_unknowns(field.type), field.name) for field in typ.members],
            defer_layout=True,
        )
    if isinstance(typ, FunctionType):
        return FunctionType(
            _fill_unknowns(typ.return_type),
            [
                (param_t if isinstance(param_t, FunctionType.VariadicParameter) else _fill_unknowns(param_t), name)
                for param_t, name in typ.parameters
            ],
        )
    return typ

def infer_types(
    function: Function[VarInstruction],
    *,
    strict: bool = True,
    include_known_constraints: bool = False,
    infer_variadic_functions: bool = False,
) -> TypeInferenceResult:
    """Infer and materialize types for unknown globals and callees.

    The intended pipeline is:

    1. run ``faultless.analysis.infer_types`` once, if it can make progress;
    2. call this function to recover external declarations;
    3. run ``faultless.analysis.infer_types`` again so local rules can reuse the
       recovered declarations.

    The pass is also robust when called before the first normal pass.  That is
    useful for fragments such as ``y = *p`` where the old local pass cannot make
    progress until external pointer evidence has been solved.

    By default, constraints whose participating types are already fully known
    are not collected.  They cannot improve external declarations, and local
    type deduction owns diagnostics for fully known expressions.  Set
    ``include_known_constraints`` to preserve the older all-constraints
    behavior.

    Set ``infer_variadic_functions`` to treat mixed-arity calls to the same
    unknown callee as evidence for a variadic function type with the common
    fixed prefix.  It is off by default so genuinely inconsistent calls remain
    visible.
    """

    solver = _TypeInferenceSolver(
        function,
        strict=strict,
        include_known_constraints=include_known_constraints,
        infer_variadic_functions=infer_variadic_functions,
    )
    _collect_constraints(solver)
    solver.solve()
    return solver.materialize()
