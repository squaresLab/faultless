"""Symbolically executes Faultless IR.
"""

from collections import deque
from dataclasses import dataclass

import z3

from .ir import *
from .analysis import find_loops


@dataclass
class EquivalenceOptions:
    """These options control how equivalence is computed, and implicitly, how execution is performed."""
    ignore_mixed_return_behavior: bool = False
    ignore_extra_arguments: bool = False
    memory_formatting_from_index: Literal[0, 1] | None = None # Default: don't assume memory formatting from either element.
    pair_names: tuple[str, str] = ("left", "right")
    use_pass_by_reference_symvars: bool = False
    infer_stack_initializer_functions: bool = True
    decompose_compound_values_in_parameter_lists: bool = False
    require_exact_function_names: bool = False # off by default because this isn't usually the case in either intended use case for faultless.

class InterpreterState:
    """Represents the interpreter state at the start or end of executing a basic block."""

    def __init__(self, stack: Stack, heap: Heap, condition: PathCondition):
        self.stack = stack
        self.heap = heap
        self.condition = condition

    def unpack(self) -> tuple[Stack, Heap, PathCondition]:
        return self.stack, self.heap, self.condition
    
class FunctionCallSymbolFactory():
    """This class builds symbolic variables treating function calls as uninterpreted functions.
    Each semantically equivalent argument list is assigned the same symbolic variable.
    """

    def __init__(self, literal_limit: int):
        """
        :param literal_limit: the maximum number of characters in an argument expression before it's replaced with a generic placeholder.
        :type literal_limit: int
        """
        assert literal_limit >= 0, f"Cannot have negative literal limit"
        self.fn2args: dict[str, list[tuple[Value, str]]] = {}
        self.literal_limit = literal_limit

    def _argument_types(self, function_type: FunctionType | None) -> Iterable[CType | None]:
        """Iterate over the types of the parameters to a function. When a variadic parameter is encountered,
        an unlimited stream of Nones are returned. The same is true when no function type is provided.
        """
        if function_type is None:
            while True:
                yield None
        else:
            for param_t, _ in function_type.parameters:
                if isinstance(param_t, FunctionType.VariadicParameter):
                    while True:
                        yield None
                else:
                    yield param_t

    def get_symvar(self, function_name: str, arguments: list[Value], function_type: FunctionType | None) -> tuple[AddressableValue[Symbol] | CompoundValue | VoidValue, list[str]]:
        """Build the symbolic variable corresponding to this function call. A function call is treated as an uninterpreted function;
        each combination of function name/argument combination is assigned a unique symbolic variable.
        """
        if function_name not in self.fn2args:
            self.fn2args[function_name] = []
        recorded_arguments = self.fn2args[function_name]

        argnames: list[str] = []
        for arg, arg_t in zip(arguments, self._argument_types(function_type)):
            # CompoundValues' types cannot be cast, and they may have type T [len]. This means they will raise an exception
            # if passed to a function argument of type T *. We special-case this here with the parenthetical in the condition,
            # because it is legal to pass an array to a function which expects a pointer.
            if arg_t is not None and not (isinstance(arg_t, Pointer) and isinstance(arg.type, Array) and arg_t.target_type == arg.type.element_type):
                arg = arg.cast(arg_t)
            for recarg, name in recorded_arguments:
                # If the cast of 'arg' succeeds, these will have identical types by construction.
                if arg.type == recarg.type and equivalent_values(arg, recarg):
                    argnames.append(name) # Use the existing name for this argument. (Each equivalent argument should map to the same name.)
                    break
            else: # on the for loop. No equivalent argument was found. Add a new one.
                argname = str(arg) if isinstance(arg, VirtualValue) else str(arg.expr)
                if len(argname) > self.literal_limit or "\n" in argname:
                    argname = f"\\carg{len(recorded_arguments)}"
                argnames.append(argname)
                recorded_arguments.append((arg, argname))
        
        symvar_name = function_name + "(" + ", ".join(argnames) + ")"
        assert isinstance(function_type, FunctionType)
        symvar_type = function_type.return_type
        if isinstance(symvar_type, Void):
            return VoidValue(), argnames
        if isinstance(symvar_type, Struct):
            return CompoundValue.make((symvar_type, symvar_name)), argnames
        return Value.make((symvar_type, symvar_name)), argnames
    
def global_variables(fn: Function[SSAInstruction]) -> set[GlobalVariable]:
    """Find and return all global variables within the provided function"""
    globals: set[GlobalVariable] = set() # the same global could be used multiple times as a storage location or operand.
    for bb in fn:
        for instruction in bb:
            if isinstance(instruction.storage, GlobalVariable):
                globals.add(instruction.storage)
            for operand in instruction.operands:
                if isinstance(operand, GlobalVariable):
                    globals.add(operand)
    return globals

def flatten_type(parameter_type: CType) -> Iterable[tuple[int, PrimitiveType | Pointer]]:
    stack: deque[tuple[int, CType]] = deque()
    stack.append((0, parameter_type))
    while stack:
        offset, current = stack.pop()
        match current:
            case PrimitiveType() | Pointer():
                yield (offset, current)
            case Struct(offset2field=o2f):
                # The way offset2field is computed the fields should be in order but by doing the sort we don't rely on that.
                stack.extend(sorted(((offset + o, f.type) for o, f in o2f.items()), key=lambda x: x[0], reverse=True))
            # Arrays can't be a type in a parameter list directly (they'll just be interpreted as a pointer),
            # but they can be nested inside a struct.
            case Array(element_type=t, nelements=n):
                assert isinstance(t, ObjectType)
                t_size = t.get_size()
                stack.extend(reversed([(offset + t_size * i, t) for i in range(n)]))
            case _:
                raise SemanticError(f"Invalid type in parameter list: {current}")

@dataclass
class FunctionProgenitorSignature:
    """A structured way of storing the symbolic variables a function generates."""
    return_value: Value
    arguments: list[Value | None]


class Execution:
    """Upon construction, interprets the function. The resulting execution object contains attributes with data about the execution."""

    def __init__(self, fn: Function[SSAInstruction], 
                 memory_symbol_factory: MemorySymbolFactory | None = None,
                 equivalence_options: EquivalenceOptions = EquivalenceOptions(),
                 differentiator: str = "",
                 use_param_names_as_symbols: bool = False
                ):
        # Function information
        self.fn = fn
        self.ins2bb = {ins: bb for bb in fn for ins in bb}
        self.loops = {loop.back_edge: loop for loop in find_loops(fn)} # indexed by back-edge because back edges (not loop heads) uniquely identify natural loops.
        self.merged_loops: dict[BasicBlock[SSAInstruction], set[BasicBlock[SSAInstruction]]] # initialized by _init_loop_exits(). Map from loop heads to bodies.
        self.loop_exit_branches = self._init_loop_exits()

        if memory_symbol_factory is None:
            memory_symbol_factory = MemorySymbolFactory()

        # Config
        self.options = equivalence_options
        assert not use_param_names_as_symbols or equivalence_options.decompose_compound_values_in_parameter_lists, f"Conflicting options: use_param_names_as_symbols and decompose_compound_values_in_parameter_lists"

        # Execution state
        self.ready: deque[tuple[BasicBlock[SSAInstruction], InterpreterState]] = deque()
        self.block_poststate: dict[BasicBlock[SSAInstruction], InterpreterState | tuple[InterpreterState, InterpreterState | None]] = {}
        self.call_factory = FunctionCallSymbolFactory(literal_limit=15)
        self.loop_head_entry_conditions: dict[BasicBlock, PathCondition] = {}
        self.back_edge_conditions: dict[BasicBlock, PathCondition] = {}
        self.registers: dict[SSAInstruction, Value | tuple[AddressableValue | None, Value]] = {}

        # Information tracked for use in equivalence checking
        self.return_states: list[tuple[PathCondition, Value | None, Stack, Heap]] = []
        self.return_constraints: list[z3.BoolRef] = []
        self.calls: dict[tuple[SSAInstruction, bool], tuple[str | Value, list[Value], Stack, Heap]] = {} # key is tuple[call_instruction, first_exec]. Value is (fname, args, stack, heap)
        self.call_vars: dict[tuple[SSAInstruction, bool], FunctionProgenitorSignature] = {} # tracks the symbolic variable names associated with each call.
        self.call_conditions: dict[SSAInstruction, PathCondition] = {} # tracks the path conditions under each call. Under the execution model, calls at loop heads should always have the same path condition, so we don't need to differentiate on first_exec
        self.loop_phi_arguments: dict[tuple[SSAInstruction, bool], Value] = {} # Note that there is only one value summarizing ALL incoming dataflow edges from inside the loop, so we have a single value on both the first and second iteration.

        ### Write parameters to the stack.
        stack = Stack(symbol_factory=memory_symbol_factory)
        arguments: list[AddressableValue[Symbol] | CompoundValue] = []
        parameter_types = [p.type for p in fn.parameters]
        i = 0
        for parameter in fn.parameters:
            # It is possible to pass arrays in C, but this is actually just an alias for passing a pointer.
            # The compiler changes any arguments that array to pointers in function parameter lists for consistency.
            assert not isinstance(parameter.type, Array)
            parameter_types = flatten_type(parameter.type) if self.options.decompose_compound_values_in_parameter_lists else ((0, parameter.type),)
            for offset, alloc_t in parameter_types:
                parameter_name = parameter.name if use_param_names_as_symbols else f"\\{differentiator}param{i}"
                value = self._default_value(alloc_t, parameter_name)
                arguments.append(value)
                stack.write(parameter, Offset(offset, True, alloc_t.get_size()), value)
                i += 1

        self.arguments = arguments
        self.rodata = ROData() # we only need one of these because it's immutable.

        # Write global variables to the stack.
        # In a normal binary, globals are not stored in the stack because they must presist across all stack frames.
        # However, faultless executes only one function at once, so the one stack frame presists for the duration of the execution.
        # Additionally, faultless assumes no particular stack layout nor amount of padding between variables in the stack fram
        # (in effect granting unlimited padding around each variable.) Thus, it is not even the case that writing globals to the stack
        # will cause any issues with respect to stack layout. Additionally, this choice makes it easier to read and write values
        # to memory: instead of tracking both a Stack and GlobalData subclass of AddressMapping and routing memory interactions
        # through the appropriate structure, we do need only to maintain a Stack and write all requests through it.
        global_progenitors: dict[Symbol, GlobalVariable] = {}
        self.global_variables: list[GlobalVariable] = []
        for gvar in global_variables(fn):
            value = self._default_value(gvar.type, f"\\global_{gvar.name}")
            stack.write(gvar, Offset(0, True, gvar.type.get_size()), value)
            if isinstance(value, AddressableValue):
                global_progenitors[value.base_address] = gvar
            else:
                for subvar in value.decompose().values():
                    assert isinstance(subvar, AddressableValue) and isinstance(subvar.base_address, Symbol)
                    global_progenitors[subvar.base_address] = gvar

        heap = Heap(symbol_factory=memory_symbol_factory)
        state = InterpreterState(stack, heap, PathCondition())
        self.ready.append((fn.entry_block, state))
        self.execute()
        self.return_value: Value | None = None
        self.return_stack: Stack # useful for looking at the final state of globals, which are stored on the stack in Faultless.
        self.return_heap: Heap # will be initialized in finalize_return_state
        self.finalize_return_state()

        # Globals are also a part of function IO/observable behavior. Finalize and record the state of all globals as well
        # as all symbols derived from those globals.
        self.globals: dict[str, Symbol] = {} # Global symbols indexed by name.
        self.global_progenitors: dict[Symbol, GlobalVariable] = global_progenitors # Maps each global progenitor symbol to the corresponding GlobalVariable object.
        self.global_derivation: dict[str, str] = {} # Maps each global name to the root progenitor name it was derived from.
        derivation_queue: deque[tuple[Symbol, Symbol]] = deque(zip(global_progenitors, global_progenitors))
        while derivation_queue:
            root, gvar = derivation_queue.popleft()
            derivations = memory_symbol_factory.get_all_derived_symbols_for(gvar)
            self.globals[gvar.name] = gvar
            self.global_derivation[gvar.name] = root.name
            derivation_queue.extend((root, d) for d in derivations)

    def _default_value(self, variable_type: CType, value_name: str) -> AddressableValue[Symbol] | CompoundValue:
        assert isinstance(variable_type, ObjectType), f"Cannot allocate a symbol of non-object type {variable_type}."
        if isinstance(variable_type, (PrimitiveType, Pointer)):
            return Value.make((variable_type, value_name))
        elif isinstance(variable_type, Struct):
            return CompoundValue.make((variable_type, value_name))
        else:
            raise NotImplementedError(f"Initialization of parameters with type {variable_type.declaration(value_name)} is currently unsupported.")

    def _init_loop_exits(self) -> dict[BasicBlock[SSAInstruction], tuple[bool, BasicBlock[SSAInstruction]]]:
        # Merge all basic blocks which share a loop head.
        merged: dict[BasicBlock, tuple[set[BasicBlock], list[tuple[bool, BasicBlock]]]] = {}
        for loop in self.loops.values():
            if loop.head in merged:
                merged[loop.head][0].update(loop.body)
            else:
                merged[loop.head] = (set(loop.body), [])
        
        # The merged loop bodies at index 0 are not modified in the code below so we can just add them directly to merged_loops without copying.
        # Initialize this here for convenience instead of in the constructor. It is declared but not initialized in the constructor.
        self.merged_loops = {head: body for head, (body, _) in merged.items()}

        # Now that the loop is merged, determine which exits are exits to the merged loop and which are internal.
        for loop in self.loops.values():
            current_merged_loop = merged[loop.head]
            for exit_block, branch in loop.exit_blocks.items():
                # branch==True means "the true branch exits", and branch==False means "the false branch exits"
                # However, the faultless convention is to put the true branch first, so we do 1-branch to check if this is still an exiting branch.
                # (It was an exiting branch on the original natural loop, but it might not be in the merged loop.)
                assert exit_block.successors[branch] in current_merged_loop[0] # the branch that wasn't an exit branch in the original natural loop should certainly not be an exiting branch in the merged loops.
                if exit_block.successors[1 - branch] not in current_merged_loop[0]:
                    current_merged_loop[1].append((branch, exit_block))
        
        exitinfo: dict[BasicBlock[SSAInstruction], tuple[bool, BasicBlock[SSAInstruction]]] = {}
        for head, (_, exit_blocks) in merged.items():
            for branch, exit_block in exit_blocks:
                blockinfo = (branch, head)
                assert exit_block not in exitinfo or exitinfo[exit_block] == blockinfo, f"Conflicting branches and/or head blocks for exit block {exit_block.id}: {exitinfo[exit_block][0]} vs {branch} and {exitinfo[exit_block][1].id} vs {loop.head.id}"
                exitinfo[exit_block] = blockinfo

        return exitinfo
    
    def perform_loop_exit(self, loop_head: BasicBlock[SSAInstruction], preexit_path: PathCondition, stack: Stack) -> PathCondition:
        """Returns a mapping from all mediloop symvars to postloop symvars."""
        assert self.is_loop_head(loop_head) # we should never call this on something that's not a loop head so this is a good sanity check. 
        mapping = []
        for instruction in loop_head:
            if not isinstance(op := instruction.op, Phi):
                break
            if op.loop_base_case is not None:  # is a loop phi
                mapping.append((op.mediloop_symvar, op.postloop_symvar))
        
        if len(mapping) == 0: # No loop-phis found. Nothing to do.
            return preexit_path
        
        preexit_condition = preexit_path.expr()
        postexit_path = preexit_path.substitute_variables(mapping)
        postexit_condition = postexit_path.expr()

        # Convert values written in mediloop symvars on the stack to the corresponding postloop symvars.
        for instruction in loop_head:
            if not isinstance(op := instruction.op, Phi):
                break
            if op.loop_base_case is not None:
                current_value = stack.read(op.variable, Offset(0, preexit_condition, op.type.get_size()), op.type)
                if isinstance(current_value, AddressableValue):
                    current_value = AddressableValue(
                        current_value.type, 
                        substitute_z3_expr(current_value.expr, mapping),
                        op.postloop_value.base_address, # the base address is for the postloop symvar
                        current_value.fields
                    )
                elif isinstance(current_value, ConditionalValue):
                    current_value = ConditionalValue(condition=substitute_z3_expr(current_value.condition, mapping))
                else:
                    # We currently don't support loop-phi composite variables, so we should not trip this assertion
                    assert type(current_value) is Value
                    current_value = Value(current_value.type, substitute_z3_expr(current_value.expr, mapping), current_value.field)
                stack.write(op.variable, Offset(0, postexit_condition, op.type.get_size()), current_value)

        return postexit_path
    
    def is_loop_head(self, block: BasicBlock[SSAInstruction]) -> bool:
        return block in self.merged_loops # indexed by head
    
    def is_loop_branch(self, block: BasicBlock[SSAInstruction]) -> bool:
        return len(block.instructions) > 0 and isinstance(block.instructions[-1].op, LoopOp)
    
    def clear_poststate_for_multiblock_head(self, head_block: BasicBlock[SSAInstruction]):
        """Deletes the post-state for all blocks in a loop head, excluding the poststate for the loop operation itself.
        
        Some loop heads are actually split into multiple basic blocks due to control-flow-inducing expressions like && and ||.
        All such blocks are in between the nautral loop head and basic block with the loop operation. This method resets their 
        poststates so that poststates from the first execution are not used in prepare_successor.
        """
        # If we have a merge of a secondary expression [as in ((x && y) || z)] then we might visit that merge block twice and thus my have already deleted the poststate.
        if head_block in self.block_poststate:
            for instruction in head_block:
                if not isinstance(instruction.op, (ControlFlowOperation, Phi)):
                    del self.registers[instruction]
            if not self.is_loop_branch(head_block):
                del self.block_poststate[head_block]
                for successor in head_block.successors:
                    self.clear_poststate_for_multiblock_head(successor)

    def read_stack_value(self, variable: Variable, stack: Stack, condition: z3.BoolRef | bool) -> Value:
        # If the variable is on the stack, the value on the stack could have been changed by reference. Thus, we
        # don't use the value in the register but instead read directly from the stack.
        if not isinstance(variable.type, ObjectType):
            raise ExecutionError(f"Currently unsupported: Reading non-object type {variable.type} (variable: {variable}) from the stack.")
        if isinstance(variable.type, Array):
            return Value.make(variable)
        else:
            read_offset = Offset(0, condition, variable.type.get_size())
            if isinstance(variable.type, Struct): # This is an optimization because reading an entire struct is expensive and often not necessary, usually we want to read just one field.
                return LazyCompoundValue(variable.type, stack.copy(), variable, read_offset)
            else:
                return stack.read(variable, read_offset, variable.type)
            
    def execute_call(self, call: FunctionCall, arguments: list[Value], heap: Heap, stack: Stack, condition: z3.BoolRef | bool) -> FunctionProgenitorSignature:
        """Implemented in place of FunctionCall.execute().
        
        We put this functionality here instead of in FunctionCall because the implementation requires several 
        attributes which logically belong in Execution.
        """
        fname = call.fname
        if isinstance(fname, Variable):
            fname = "%" + fname.name # don't use the & sigil here because this isn't really taking the address.
        elif isinstance(fname, SSAInstruction):
            fname = fname.out_repr if fname.out_repr is not None else "%" + str(id(fname))

        result_value, argnames = self.call_factory.get_symvar(fname, arguments, call.ftype)
        
        argvars: list[Value | None] = []
        assert len(argnames) == len(arguments)

        # Write pass-by-reference values to the stack (and possibly heap) if configured to do so.
        for i, argument in enumerate(arguments):
            if isinstance(argument, AddressableValue) and not isinstance(argument.base_address, RODataAddress) and \
               (self.options.use_pass_by_reference_symvars or (self.options.infer_stack_initializer_functions and isinstance(argument.base_address, Variable))):
                if isinstance(argument.type, (Pointer, Array)):
                    write_t = MemoryOperation.type_at_address(argument)
                else:
                    write_t = INTEGER
                assert isinstance(write_t, ObjectType)
                referenced_argname = argnames[i]
                argnames[i] = "*" + referenced_argname # To indicate this variable represents this parameter for this pass-by-reference symbolic value.
                value_args = (write_t, fname + "(" + ", ".join(argnames) + ")")
                argnames[i] = referenced_argname

                value_to_write = CompoundValue.make(value_args) if isinstance(write_t, (Struct, Array)) else Value.make(value_args)
                offset = Offset(0, condition, write_t.get_size())
                if isinstance(argument.base_address, Variable):
                    stack.write(argument.base_address, offset, value_to_write)
                else:
                    heap.write(argument.base_address, offset, value_to_write)
                # The choice to decompose here is significant. By decomposing them, we're flattening the parameter lists out (of each function, if the 
                # options are the same on each execution). This allows individual primitive/pointer 'leaf' values to align with each other across
                # parameter boundaries. So for instance, if one function had the arguments func3(x, y) where x and y were ints, and the other had the
                # call bar(pt), where pt is of type struct point { int x; int y; }, the arguments of func3 and align with the fields of bar.
                # The downstream code in the product program builder/synchronizer is designed to respect the values provided here, so decomposing here
                # (as intended for decompiled code) helps create that cross-parameter positional alignment effect.
                if self.options.decompose_compound_values_in_parameter_lists and isinstance(value_to_write, CompoundValue):
                    argvars.extend(value_to_write.decompose().values())
                else:
                    argvars.append(value_to_write)
            else:
                argvars.append(None)
        return FunctionProgenitorSignature(result_value, argvars)

    def execute_block(self, block: BasicBlock[SSAInstruction], in_state: InterpreterState):
        """Inner interpreter loop (instruction level). Sequentially executes each instruction in a single block."""
        stack, heap, path_condition = in_state.unpack()

        block_path_condition = [] # the path condition can contain not just the value passed to the branch statement but also constraints added by loop-phi nodes.
        instruction = None # Just to have the variable "instruction" bound to something if the basic block contains no instructions.
        first_exec = (block not in self.block_poststate) # Loop heads are executed twice: to start and to complete the induction.
        is_loop_head = self.is_loop_head(block)
        condition = path_condition.expr()

        if is_loop_head:
            if first_exec:
                self.loop_head_entry_conditions[block] = path_condition
            else:
                self.clear_poststate_for_multiblock_head(block) # is a no-op for a single-block head.
                back_edge_merge_condition = condition # We'll need this to read into loop-phis properly.
                self.back_edge_conditions[block] = path_condition # Save this for proving phi-instructions equivalent
                # Don't carry the complexities of the in-loop path constraints out the loop exit; they don't matter there.
                # To accomplish this, on the second execution of a loop head, overwrite the path condition with the path condition
                # from the first time the loop was executed.
                path_condition = self.loop_head_entry_conditions[block]
                condition = path_condition.expr()

        for instruction in block:
            ### Prepare the arguments to the instruction.
            arguments: list[Value] = []
            # If the instruction is a Phi, the execution doesn't actually involve the arguments.
            # For non-loop Phis, it involves reading from the stack at the storage location (variable) that the phi instruction corresponds to.
            # Loop phis need to access the memory before the memory merge for this block, so they need their own special argument-reading code.
            if not isinstance(instruction.op, Phi):
                for operand in instruction.operands:
                    if isinstance(operand, (Parameter, GlobalVariable)) or isinstance(operand, Uninitialized):
                        if isinstance(operand, Uninitialized):
                            operand = operand.value
                        arguments.append(self.read_stack_value(operand, stack, condition))
                    elif isinstance(operand, SSAInstruction):
                        if isinstance(operand.storage, Variable):
                            arguments.append(self.read_stack_value(operand.storage, stack, condition))
                        else:
                            stored = self.registers[operand]
                            if isinstance(stored, tuple):
                                if isinstance(instruction.op, (SizeOf, AddressOf)) and stored[0] is not None and isinstance(stored[0].type, Array):
                                    assert isinstance(stored[1], Pointer) and stored[1].target_type == stored[0].type.element_type
                                    # The rval in a register is the right element (at position 1). In most cases, when used as rvals,
                                    # arrays decay into pointers, so we store the rvalue with a pointer type. This will give the incorrect
                                    # result when passed to SizeOf. So instead we use the lval, which we preserve with an array type.
                                    rval = stored[0]
                                else:
                                    rval = stored[1]
                            else:
                                rval = stored
                            arguments.append(rval)
                    elif isinstance(operand, Constant): # Uninitialized constants are handled with parameters above. Both are stack reads.
                        arguments.append(Value.make(operand))
                    elif isinstance(operand, CType) and isinstance(instruction.op, (Cast, SizeOf)) and (isinstance(instruction.op, SizeOf) or len(arguments) == 0):
                        # The first argument to a cast operation and the argument to sizeof(type) are allowed to be types.
                        arguments.append(operand) # type: ignore -- fudging the type system here.
                    else:
                        raise ValueError(f"Invalid SSAOperand type: {type(operand)} ({operand})")
            
            ### Execute the instruction and store the results.
            
            # This first case is a special case which doesn't fit the mold of a typical MemoryOperation execution very well.
            # There is no valid lval when reading from a string literal, and most MemoryOperations don't need to take an ROData.
            if isinstance(instruction.op, (Subscript, Dereference)) and isinstance(arguments[0], AddressableValue) and isinstance(arguments[0].base_address, RODataAddress):
                char_type = arguments[0].base_address.literal.type.element_type
                assert isinstance(char_type, Integer)
                assert isinstance(arguments[0].type, Array) and arguments[0].type.element_type == char_type or \
                       isinstance(arguments[0].type, Pointer) and arguments[0].type.target_type == char_type, f"Base address for a string literal should be a {char_type} [] or {char_type} * but found {arguments[0]}"
                index = arguments[0].compute_offset()
                if isinstance(instruction.op, Subscript):
                    char_ptr_type = Pointer(char_type)
                    index = index + char_type.get_size() * arguments[1].cast(char_ptr_type).expr
                # It is important that we set the value of result_value on each execution path.
                result_value = self.rodata.read(arguments[0].base_address, Offset(index, condition, char_type.get_size()), char_type)
                self.registers[instruction] = result_value
            elif isinstance(instruction.op, MemoryOperation):
                lval_operand = instruction.operands[0] # Conveniently, the way the syntax works out, it's always the first operand.
                # TODO: support non-parameter local arrays (currently these are just an "uninitialized" constant.)
                if lval_operand in self.registers and isinstance(self.registers[lval_operand], tuple):
                    lval = self.registers[lval_operand][0] # type: ignore
                elif isinstance(lval_operand, Variable):
                    lval = Value.make(lval_operand)
                elif isinstance(lval_operand, SSAInstruction) and lval_operand.storage is not None:
                    lval = Value.make(lval_operand.storage)
                elif isinstance(lval_operand, Uninitialized):
                    variable = lval_operand.value
                    # Uniquely among C types, the value of an array variable and its address (as returned by &) are the same thing.
                    # That is if 'arr' is an array, arr and &arr are the same value (i.e. the same memory address). However, they
                    # are of different types: arr is of type element_type, while &arr is of type Pointer(Array(element_type, nelements)).
                    # We simulate this by passing a new Variable with the same name but new type to Value.make().
                    if isinstance(variable.type, Array):
                        # In general, because Variables compare and hash by id (memory address), we do not create copies of them.
                        # However, Value.make() processes one Variable at a time and does not copy any variables. Therefore this is safe.
                        variable = Variable(Pointer(variable.type), variable.name, variable.is_temporary, variable.is_stack_allocated)
                    lval = Value.make(variable)
                    del variable
                else:
                    lval = None

                # AddressableValues with RODataAddresses are a special case and are handled manually above, not in a general-purpose MemoryOp.
                assert lval is None or isinstance(lval.base_address, (Symbol, Variable)), f"Memory operations should handle only AddressableValues with Symbol or Variable base addresses."
                assert not isinstance(arguments[0], AddressableValue) or isinstance(arguments[0].base_address, (Symbol, Variable)), f"Memory operations should handle only AddressableValues with Symbol or Variable base addresses."

                # It is important we unpack the return value of execute here to set result_value.
                lval, result_value = instruction.op.execute(arguments, lval=lval, stack=stack, heap=heap, condition=condition) # type: ignore -- the multiple-inheritence-based method of subclassing MemoryOperation to recognize which operations need the additional kwargs does not work with the typechecker.
                self.registers[instruction] = (lval, result_value)
            elif isinstance(instruction.op, Phi):
                if instruction.op.loop_base_case is not None:
                    # A loop-phi instruction needs only one argument. There is only one base case because there is only one entry point
                    # to the loop. Unlike other operations, loop-phis are stateful. They'll remember the base case.
                    # The stack implements the path-merging logic we need so 
                    phi_storage = instruction.op.variable
                    assert isinstance(phi_storage.type, ObjectType), f"Currently unsupported: non-object type {phi_storage.type} (variable: {phi_storage}) for the loop-phi {instruction}."
                    
                    query = Offset(0, condition if first_exec or not is_loop_head else back_edge_merge_condition, phi_storage.type.get_size())
                    argument = stack.read(phi_storage, query, phi_storage.type)
                    # TODO: propagate inductive_info to values derived from this value. Right now it is not tracked at all.
                    result_value, inductive_info = instruction.op.execute([argument], first_exec=first_exec, block_path_condition=block_path_condition, heap=heap)
                    self.registers[instruction] = result_value
                    
                    # Record the argument for use in the proving step. The base case will be on the first execution and the inductive step input will be on the second execution.
                    self.loop_phi_arguments[(instruction, first_exec)] = argument
                else:
                    # We don't evaluate the arguments and the combine them together with a z3.If expression because this is already
                    # implicitly done in the stack. Instead, we can simply read directly from the stack.
                    storage = instruction.op.variable
                    assert storage.is_stack_allocated, f"Phi instructions should not have temporary non-stack-allocated storage."
                    assert all((operand == storage if isinstance(operand, Variable) else (operand.value if isinstance(operand, Uninitialized) else operand.storage == storage)) for operand in instruction.operands), f"Inconsistent storage for phi instruction {instruction}" # type: ignore
                    assert isinstance(storage.type, ObjectType), f"Can't store non-object type {storage.type}"
                    self.registers[instruction] = stack.read(storage, Offset(0, condition, storage.type.get_size()), storage.type)
            elif isinstance(instruction.op, ControlFlowOperation):
                break
            elif isinstance(instruction.op, FunctionCall):
                if instruction in self.call_conditions:
                    assert unsatisfiable(condition != self.call_conditions[instruction].expr()), f"Nonequivalent path conditions on second execution of {instruction}: {self.call_conditions[instruction]} vs {condition}"
                
                function_name = instruction.op.fname
                if isinstance(function_name, SSAInstruction):
                    if function_name.storage is not None:
                        function_name = function_name.storage # then we'll execute the if isinstance(function_name, Variable) immediately below.
                    else:
                        function_name = self.registers[function_name]
                        if isinstance(function_name, tuple):
                            function_name = function_name[1] # the rvalue
                if isinstance(function_name, Variable):
                    function_name = self.read_stack_value(function_name, stack, condition)

                # If requested, flatten compound values into a sequence of atomic values.
                save_args = []
                for arg in arguments:
                    if self.options.decompose_compound_values_in_parameter_lists and isinstance(arg, CompoundValue):
                        save_args.extend(arg.decompose().values())
                    # We don't decompose string literals, although they are in fact compound values. This is because of the way the proof algorithm
                    # handles string arrays (including string literals, which are immutable arrays). The actual value passed to the function in C code
                    # when passing an array is just a pointer to that array. Faultless' strategy for pass-by-reference arguments on the stack (like 
                    # arrays) is to compare the memory in the array itself rather than the pointer. Likewise, for string literals, we compare the value
                    # in the array. We read array memory in prover.py, but to keep .rodata localized to the interpreter we read string literals here.
                    # (In effect, this treats string literals more like constant values than pointers.)
                    elif isinstance(arg, AddressableValue) and isinstance(arg.base_address, RODataAddress):
                        if not unsatisfiable(arg.expr != arg.base_address.symvar):
                            raise NotImplementedError(f"Support for pointers to partial strings is not implemented.")
                        save_args.append(self.rodata.get_string_value(arg.base_address))
                    else:
                        save_args.append(arg)

                # Record information needed to prove calls equivalent. This is information that could impact the call: how it behaves or whether or not it is executed.
                self.calls[(instruction, first_exec)] = (function_name, save_args, stack.copy(), heap.copy()) # make a copy: record the state of memory excatly as it is at the time of the function call.
                self.call_conditions[instruction] = path_condition

                # Don't send the potentially flattened argument list to execute_call so that when we allocate the function symvar we can reliably compare each argument
                # with the type of the corresponding parameter type in the function signature.
                call_values = self.execute_call(instruction.op, arguments, heap, stack, condition)
                
                # Save call information to local execution state.
                result_value = call_values.return_value # It is important that we set the value of result_value on each code path
                self.call_vars[(instruction, first_exec)] = call_values
                self.registers[instruction] = result_value
            else:
                # It is important that we set the value of result_value on each code path
                result_value = instruction.op.execute(arguments)
                self.registers[instruction] = result_value

            # Store the result of executing the instruction in the stack if necessary.
            # Note that we don't want to store the results of executing non-loop-phi instructions on the stack, since these values
            # come from reading the stack in the first place; writing them back would be redundant.
            if instruction.storage is not None and not (isinstance(instruction.op, Phi) and instruction.op.loop_base_case is None):
                assert isinstance(instruction.storage.type, ObjectType), f"Cannot store non-object type {instruction.storage.type} (in instruction {instruction})"
                stack.write(instruction.storage, Offset(0, condition, instruction.storage.type.get_size()), result_value.cast(instruction.storage.type))

        ### post-execution. Clean up and prepare next blocks.
        if instruction is not None and isinstance(instruction.op, ControlFlowOperation):
            assert instruction == block.instructions[-1], f"Control flow instruction is not the last instruction in the basic block." # Sanity check
            # Compute branch condition
            branch_condition: z3.BoolRef | bool | None = None
            if isinstance(instruction.op, (If, LoopOp)): # breaks and continues don't have branches.
                condition_val = arguments[0]
                if isinstance(condition_val, ConditionalValue):
                    branch_condition = condition_val.condition
                elif isinstance(condition_val.type, (PrimitiveType, Pointer)):
                    branch_condition = truthiness(condition_val)

            # branch condition is set. Used in processing If and LoopOp below.
        # We should only get these components on the first execution of a loop head, and may not get them even then (depending on the loop).
        assert len(block_path_condition) == 0 or (instruction is not None and is_loop_head and first_exec), f"Unexpected extra components in the path condition for block {block.id} on execution {first_exec + 1}: {block_path_condition}"
        
        # Handle control flow:
        operation = instruction.op if instruction is not None else None
        if isinstance(operation, If):
            assert len(arguments) == 1, f"Incorrect number of arguments to if instruction: {arguments}"
            assert branch_condition is not None, f"Nonscalar type can't be used to determine branching behavior."
            true_condition, false_condition = path_condition.branch(block, branch_condition)
            true_stack = stack
            false_stack = stack
            if block in self.loop_exit_branches:
                exit_is_true_branch, exiting_loop_head = self.loop_exit_branches[block]
                if exit_is_true_branch:
                    true_stack = true_stack.copy()
                    true_condition = self.perform_loop_exit(exiting_loop_head, true_condition, true_stack)
                else:
                    false_stack = false_stack.copy()
                    false_condition = self.perform_loop_exit(exiting_loop_head, false_condition, false_stack)

            self.block_poststate[block] = (InterpreterState(true_stack, heap, true_condition), InterpreterState(false_stack, heap, false_condition))
            assert len(block.successors) == 2
            self.prepare_successor(block.successors[0])
            self.prepare_successor(block.successors[1])
        elif isinstance(operation, LoopOp):
            assert branch_condition is not None, f"Nonscalar type can't be used to determine branching behavior."
            if first_exec:
                block_path_condition.append(branch_condition)
                loop_invariant: z3.BoolRef | None = None
                if len(block_path_condition) > 1:
                    loop_invariant = z3.And(*block_path_condition) # type: ignore -- z3 typing
                elif len(block_path_condition) == 1:
                    loop_invariant = block_path_condition[0]
                true_condition, _ = path_condition.branch(block, branch_condition, loop_invariant)
                self.block_poststate[block] = (InterpreterState(stack, heap, true_condition), None)
                self.prepare_successor(block.successors[0])
            else:
                _, false_condition = path_condition.branch(block, branch_condition)
                first_exec_poststate = self.block_poststate[block]
                assert isinstance(first_exec_poststate, tuple) and isinstance(first_exec_poststate[0], InterpreterState) and first_exec_poststate[1] is None, f"Unexpected existing post-state for second loop head execution."
                self.block_poststate[block] = (first_exec_poststate[0], InterpreterState(stack, heap, false_condition))
                self.prepare_successor(block.successors[1])
        elif isinstance(operation, Return):
            self.block_poststate[block] = InterpreterState(stack, heap, path_condition)
            retval = arguments[0] if len(arguments) > 0 else None
            self.return_states.append((path_condition, retval, stack, heap))
        else:
            self.block_poststate[block] = InterpreterState(stack, heap, path_condition)
            assert len(block.successors) <= 1,f"Only basic blocks ending in a branch instruction can have more than one successor: {block}" 
            if len(block.successors) == 1:
                self.prepare_successor(block.successors[0])

        # Account for block which result in a natural termination of execution (without a return statement).
        if len(block.successors) == 0 and not isinstance(operation, Return):
            self.return_states.append((path_condition, None, stack, heap))

    def prepare_successor(self, block: BasicBlock[SSAInstruction]):
        in_states: list[InterpreterState] = []
        already_executed = block in self.block_poststate
        is_loop_head = self.is_loop_head(block)
        assert not already_executed or (is_loop_head or self.is_loop_branch(block))
        for predecessor in block.predecessors:
            # If the block is a loop head, ignore predecessors that are back edges.
            # Otherwise, only include back edges.
            if is_loop_head and (already_executed != ((predecessor, block) in self.loops)):
                continue
            if predecessor not in self.block_poststate:
                # This means that not all predecessors' poststates have been computed, so we cannot prepare the prestate for this block.
                # This function will be called again each time one of this block's predecessors is executed, so this if will eventually be false.
                # (Back edges from loops are ignored and gotos are not supported so there are no cycles.)
                return
            if len(predecessor.successors) == 1:
                assert predecessor.successors[0] == block
                poststate = self.block_poststate[predecessor]
                assert isinstance(poststate, InterpreterState) # As opposed to a tuple of interpreter states from a branch situation.
                in_states.append(poststate)
            else:
                assert len(predecessor.successors) == 2, f"A predecessor basic block to another block should have exactly one or two successors but found {len(predecessor.successors)}"
                is_from_true_branch = predecessor.successors[0] == block
                is_from_false_branch = predecessor.successors[1] == block
                assert is_from_true_branch != is_from_false_branch, f"A basic block that is the successor of another block with two successors must be exactly one of the true or false successor but got {is_from_false_branch} and {is_from_true_branch}"
                poststate = self.block_poststate[predecessor]
                assert isinstance(poststate, tuple)
                poststate = poststate[is_from_false_branch] # will be interpreted as 0 when it's the true branch and 1 when it's the false branch
                if poststate is None:
                    return # This must be a loop for which we haven't executed the false branch yet.
                in_states.append(poststate)

        # Merge the incoming states.
        if len(in_states) >= 2:
            condition, merge = PathCondition.merge([s.condition for s in in_states])
            stack = merge_memory([s.stack for s in in_states], merge)
            heap = merge_memory([s.heap for s in in_states], merge)
        else:
            assert len(in_states) == 1
            stack, heap, condition = in_states[0].unpack()
            # PathConditions are immutable so we don't need to copy those.
            stack = stack.copy()
            heap = heap.copy()

        state = InterpreterState(stack, heap, condition)
        self.ready.append((block, state)) 

    def execute(self):
        """Outer interpreter loop (basic-block level)."""
        while len(self.ready) > 0:
            self.execute_block(*self.ready.popleft())

    def finalize_return_state(self):
        if len(self.return_states) == 1:
            path_constraint, rv, stack, heap = self.return_states[0]
            if rv is not None:
                self.return_value = rv.cast(self.fn.return_type) # TODO: handle path constraints.
            self.return_stack = stack.copy()
            self.return_heap = heap.copy()
            if path_constraint:
                self.return_constraints.append(path_constraint.expr())
        else:
            conditions = [state[0] for state in self.return_states]
            values = [state[1] for state in self.return_states if state[1] is not None]
            stacks = [state[2] for state in self.return_states]
            heaps = [state[3] for state in self.return_states]
            base_constraint, merge = PathCondition.merge(conditions)

            if base_constraint:
                self.return_constraints.append(base_constraint.expr())

            ### Return value
            def build_expr(node: PathCondition.MergeTree | int) -> SymbolicExpression:
                if isinstance(node, PathCondition.MergeTree):
                    return z3.If(node.decision, build_expr(node.true), build_expr(node.false)) # type: ignore
                else:
                    return cast(values[node].expr, values[node].type, self.fn.return_type)
            
            if isinstance(self.fn.return_type, Void):
                if len(values) > 0:
                    raise SemanticError("Return values provided for a void function: " + ', '.join(str(v) for v in values))
            else:
                # Require one return value for each terminating block
                if len(values) == sum(len(bb.successors) == 0 for bb in self.fn.basic_blocks):
                    expr = build_expr(merge)
                    self.return_value = Value(self.fn.return_type, expr)
                else:
                    raise SemanticError("Return values were not provided along all paths.")
            
            ### Return heapstate
            self.return_stack = merge_memory(stacks, merge)
            self.return_heap = merge_memory(heaps, merge)

            ### Return constraints
            def visit_edge(child: PathCondition.MergeTree | int, asserts: ComponentwisePathT, components: list[z3.BoolRef | bool]):
                """If this edge is constrained (that is, all paths containing these components so far also included this constraint)
                we record that fact as an implication: having gone down the path of executing these components means that the constraints
                recorded on this edge also hold.
                """
                edge_constraints = [c[0] for c in asserts.values()]
                if len(edge_constraints) > 0:
                    self.return_constraints.append(z3.Implies(z3.And(*components), z3.And(*edge_constraints)))
                components.extend(edge_constraints)
                visit_node(child, components)
                if len(edge_constraints) > 0:
                    del components[-len(edge_constraints):]

            def visit_node(node: PathCondition.MergeTree | int, components: list[z3.BoolRef | bool]):
                """Adds the decision node to the current path and explores down each subtree."""
                if isinstance(node, int):
                    return
                components.append(node.decision)
                visit_edge(node.true, node.true_asserts, components)
                components.pop()

                components.append(not node.decision if isinstance(node.decision, bool) else z3.Not(node.decision)) # type: ignore
                visit_edge(node.false, node.false_asserts, components)
                components.pop()
            
            visit_node(merge, []) # Will populate the self.return_constraints list
