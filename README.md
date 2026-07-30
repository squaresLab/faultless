# Faultless
Faultless is a program equivalence technique designed to determine if two C functions are equivalent. It is primarily designed to operate on code representations that occur in the context of reverse engineering and neural decompilation: source code, deterministically decompiled code, and code produced by a neural decompiler. Neural decompilers are machine learning models that perform the task of decompilation, converting code in a low-level language (like assembly) to a higher level language (like C).

Neural decompilers can be very useful, but they have no correctness guarantees and therefore can misrepresent the underlying low level code without warning. Faultless is a python package that checks for the presence/absence of semantic mistakes in neural decompilers by comparing against a reference solution with stronger correctness properties.

Faultless is designed to check for correctness in two different scenarios by comparing with two different reference implementations:
- Validation: the neural decompiler's prediction is compared with the output of a deterministic decompiler like the IDA Pro or Ghidra decompiler, which have stronger correctness properties than neural decompilers. This mode makes sense for reverse engineers actively using neural decompilers in their toolchains, because deterministically decompiled code is often available during manual reverse engineering.
- Evaluation: the neural decompiler is used to compare with the original source code from which the binary is built. This makes sense in controlled evaluation settings where a researcher can build a dataset from source. In some senses the ideal decompiler is one which can perfectly reconstruct the source code.

Faultless is currently a proof of concept.

## Installation

` pip install .`

Faultless has very few dependencies outside of the python standard library: `tree-sitter`, a parsing library, the C-language package for tree-sitter, and `z3`, an SMT solver.

If desired, run unit tests with `python -m unittest`.

## API

The main API entry points are `are_equivalent` and `compare`. The former returns a boolean, whereas the latter provides a comparison object with richer information about the proof performed by faultless and the reason for nonequivalence, if any.

Faultless takes the functions for comparison as strings. Prepend the definitions of any types used in the function.

```py
from faultless import are_equivalent, compare, Comparison

code1 = """int inc(int x) { return x + 1; }"""
code2 = """int dec(int x) { return x - 1; }"""

equivalent: bool = are_equivalent(code1, code2)
comparison: Comparison = compare(code1, code2)

```

If you'd like to statically symbolically execute the code without performing any equivalence checking, run
```python
from faultless import execute, Execution

execution: Execution = execute(code1)
```

The `are_equivalent`, `compare`, and `execute` functions all take an `EquivalenceOptions` parameter. The symbolic representations of C types can be controlled with the `z3repr_options` context manager.

Finally, because Faultless is designed to handle decompiled code and because decompilers often introduce types outside the C type system which function as generic types for types of given size, Faultless features the `set_decompiler_placeholder_types` function, which notifies Faultless as to what these types are. There is a pre-set option `'Hex-Rays'` for Hex-Rays' IDA Pro decompiler. You can also manually pass in the types instead.


## Options

The `EquivalenceOptions` class has the following options:
- `ignore_mixed_return_behavior`: if one of the two functions returns a value and the other is `void`, ignore return behavior when determining equivalence. (default: `False`)
- `ignore_extra_arguments`: When comparing two function calls, if one has more arguments than another, ignore the excess arguments when computing equivalence. (default: `False`)
- `memory_formatting_from_index`: When comparing the contents of memory referenced by two pointers, interpret the values in memory according to the base type found in the function of the specified index. (default: `None`)
- `pair_names`: Specifies the names of the functions under comparison in error messages. (default: `('left', 'right')`)
- `use_pass_by_reference_symvars`: During symbolic execution, when a function call is made, conservatively assume that all arguments that could be memory addresses are written to by reference. Write a symbolic value at each such memory address. (default: `False`)
- `infer_stack_initializer_functions`: If a variable is uninitialized and its address is passed to a function, assume that that function initializes the variable by reference (like `scanf` does) and write a symbolic value to that variable indicating the initialization. (default: `True`)
- `decompose_compound_values_in_parameter_lists`: convert `struct` and array values in parameter lists to a sequence of primitive values. (default: `False`)
- `require_exact_function_names`: Adds as a necessary condition for determining the equivalence of two function calls the requirement that the names of the called functions are identical. (default: `False`) 

For the evaluation task (comparing a neural decompiler's prediction with the original source), the default options are a good starting point.

For the validation task (comparing the neural decompiler's prediction with that from a deterministic decompiler), the following `EquivalenceOptions` are a good starting point:

```python
EQUIVALENCE_OPTIONS = EquivalenceOptions(
    ignore_mixed_return_behavior=True,
    ignore_extra_arguments=True,
    memory_formatting_from_index=0,
    decompose_compound_values_in_parameter_lists=True,
    pair_names=("original", "decompiled"),
)
```

This configuration of the options assumes that the original code is passed in first, as in `compare(original, decompiled, equivalence_options=EQUIVALENCE_OPTIONS)`.

To set the symbolic representations of integer and floating point variables, use the `z3repr_options` context manager. For example, 

```
with z3repr_options(integer_repr='bitvec', pointer_repr='int'):
    result = are_equivalent(func1, func2)
```

The options for integers (`integer_repr`) and pointers (`pointer_repr`) are:
- `bitvec`: a bit vector. The number of bits is given by the size of the type.
- `int`: a mathematical integer.

The option for floating point numbers (`float_repr`) are:
- `float`: an IEEE-754 floating point number.
- `real`: a mathematical real number


## Architecture

Faultless has three main components:
- A compiler, which converts raw C code into an intermediate representation (faultless IR).
- A symbolic interpreter, which performs static symbolic execution on faultless IR and saves various pieces of execution state.
- A proof engine, which takes in two execution states (one from each function) and uses this information to determine if the two input functions are equivalent or not.
