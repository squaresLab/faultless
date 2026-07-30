"""Defines the main API for the package."""

from faultless.ir import Function, SSAInstruction, MemorySymbolFactory, z3repr_options, set_decompiler_placeholder_types
from faultless.c import compile
from faultless.analysis import convert_to_ssa, init_loop_phi_base_cases, deduce_types
from faultless.type_inference import infer_types
from faultless.interpreter import Execution, EquivalenceOptions
from faultless.prover import SynchronizationProof, synchronize, prove_equivalence

__all__ = [
    "are_equivalent",
    "compare",
    "Comparison",
    "execute",
    "z3repr_options",
    "EquivalenceOptions",
    "set_decompiler_placeholder_types"
]

class Comparison:
    """Stores information about the comparison of two functions."""
    def __init__(self, 
                 left: Function[SSAInstruction], 
                 right: Function[SSAInstruction], 
                 product_program: SynchronizationProof,
                 nonequivalence_reason: str | None
                ):
        self.left = left
        self.right = right
        self.product_program = product_program
        self.nonequivalence_reason = nonequivalence_reason
    
    def equivalent(self):
        return self.explain_nonequivalence() is None
    
    def explain_nonequivalence(self) -> str | None:
        return self.nonequivalence_reason
    
    def __repr__(self):
        output = [f"Comparison({self.left.name}, {self.right.name}):"]
        output.extend(repr(self.product_program).strip().splitlines()[1:]) # remove the header line and just put everything under a Comparison header.
        if self.nonequivalence_reason is None:
            output.append(f"{self.left.name} and {self.right.name} are equivalent")
        else:
            output.append(f"{self.left.name} and {self.right.name} are not equivalent: {self.nonequivalence_reason}")
        return "\n".join(output)

def compare(left: str, right: str, left_function_name: str | None = None, right_function_name: str | None = None, equivalence_options: EquivalenceOptions = EquivalenceOptions()) -> Comparison:
    """Compare two functions, returning an information package about the result of the comparison which can be used to determine if the functions are equivalent and if not, why.

    :param left: a string containing one target C function and any other supporting definitions or declarations
    :param right: a string containing the other target C function and any other supporting definitions or declarations
    :param left_function_name: specifies the function in 'left' to execute. Required if 'left' contains multiple function definitions.
    :param right_function_name: specifies the function in 'right' to execute. Required if 'right' contains multiple function definitions.
    :param equivalence_options: a configuration object that controls how equivalence is computed and output is displayed.
    """
    symbol_factory = MemorySymbolFactory()
    left_exec = execute(left, left_function_name, "left", symbol_factory, equivalence_options, "l")
    right_exec = execute(right, right_function_name, "right", symbol_factory, equivalence_options, "r")
    product_program = synchronize(left_exec, right_exec, symbol_factory, equivalence_options)
    reason = prove_equivalence(left_exec, right_exec, product_program, equivalence_options)
    return Comparison(left_exec.fn, right_exec.fn, product_program, reason)

def are_equivalent(left: str, right: str, left_function_name: str | None = None, right_function_name: str | None = None, equivalence_options: EquivalenceOptions = EquivalenceOptions()) -> bool:
    """Return True if the functions in left and right are equivalent or False if not.

    :param left: a string containing one target C function and any other supporting definitions or declarations
    :param right: a string containing the other target C function and any other supporting definitions or declarations
    :param left_function_name: specifies the function in 'left' to execute. Required if 'left' contains multiple function definitions.
    :param right_function_name: specifies the function in 'right' to execute. Required if 'right' contains multiple function definitions.
    :param equivalence_options: a configuration object that controls how equivalence is computed and output is displayed.
    """
    return compare(left, right, left_function_name, right_function_name, equivalence_options).equivalent()

def execute(
        code: str, fn_name: str | None = None, codename: str | None = None, 
        symbol_factory: MemorySymbolFactory | None = None, 
        equivalence_options: EquivalenceOptions = EquivalenceOptions(),
        differentiator: str = ""
    ) -> Execution:
    """Executes the function in the parameter 'code' or the function specified by 'fn_name'
    if there are multiple functions in fn_name.

    :param code: C code containing a function to be executed.
    :param fn_name: The name of a function in 'code' to execute. Required if 'code' contains multiple function definitions.
    :param codename: The name of this piece of code. Used to provide better error messages.
    :param symbol_factory: An object which consistently maps equivalent generated symbols to the same name across runs.
    :param equivalence_options: A configuration object that controls how equivalence is computed and output is displayed.
    :param phi_differentiator: A string incorporated into each phi-instruction and parameter name to make it distinct from 
        a phi node for a variable of the same name in the other function.
    """
    functions = compile(bytes(code, "utf8"))
    if fn_name is None:
        if len(functions) == 0:
            raise ValueError(f"There are no functions in the {codename} code.")
        elif len(functions) > 1:
            raise ValueError(f"The {codename} code has multiple functions; specify one with `fn_name`.")
        var_ir = functions[0]
    else:
        for var_ir in functions:
            if var_ir.name == fn_name:
                break
        else:
            raise ValueError(f"Function {fn_name} is not found in the provided {codename} code.")
    deduce_types(var_ir) # Get as much type info as we can with the type information we have from parameters/local declarations
    infer_types(var_ir) # Infer the types of globals and callees. Diagnostics returned here could be useful for debugging.
    deduce_types(var_ir) # Propagate inferred callee/global information throughout the rest of the function.
    ssa_ir = convert_to_ssa(var_ir, phi_differentiator=differentiator)
    if init_loop_phi_base_cases(ssa_ir): # only run the pre-execution pass if necessary. (init_loop_phi_base_cases returns the number of loop phis initialized.)
        Execution(ssa_ir, symbol_factory, equivalence_options=equivalence_options, differentiator=differentiator) # pre-execution to infer inductive step constraints
    return Execution(ssa_ir, symbol_factory, equivalence_options=equivalence_options, differentiator=differentiator) # main execution
