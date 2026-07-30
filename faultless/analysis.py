"""Program analysis code necessary for code alignment
"""
import itertools
from queue import Queue
from typing import Callable, TypeVar, Iterable
import copy

from .ir import *


def deduce_types(function: Function[VarInstruction]):
    """Update a Function in-place, replacing UnknownTypes with other CTypes in instructions' 
    result variables where possible.
    """
    for bb in function:
        for _instruction in bb:
            instruction: VarInstruction = _instruction # for typechecking
            inferred_t = instruction.op.deduce_type(*instruction.operands)
            if inferred_t is not None:
                assert instruction.result is not None, f"Instruction {instruction} should have a result variable but none found."
                if isinstance(instruction.result.type, UnknownType):
                    assert instruction.result.is_temporary or isinstance(instruction.result, GlobalVariable), f"Declared (non-temporary, non-global) variable {instruction.result} has an unknown type."
                    # Because all instances of a given variable are references to the same object,
                    # updating the type of the result variable of an expression updates the same variable
                    # in the argument lists of other instructions.
                    instruction.result.type = inferred_t
                else:
                    # For copy operations, we have to consider the case where the operand is a 0 integer literal being assigned to a pointer, but .deduce_type only returns a type.
                    # Thus, we manually extract the copy operation's operand here and pass it to assignable to allow for this case.
                    check = instruction.operands[0] if isinstance(instruction.op, Copy) else inferred_t
                    if not assignable(instruction.result.type, check):
                        raise SemanticError(f"Instruction result type {inferred_t} cannot be assigned to variable {instruction.result} of type {instruction.result.type}.")
                    # Else don't update the type with the inferred type; if it's there it was determined by
                    # variable declarations directly. We just need to check that the results can actually
                    # be stored in objects of the variables' type.
            else:
                assert instruction.result is None, f"Instruction {instruction} has a result variable, but no type was inferred for it."

#### Generic dataflow analysis framework

T = TypeVar("T") # Lattice element T

def dataflow(function: Function, 
             transfer_fn: Callable[[list[T], BasicBlock], list[T]], # Note: the output list[T] MUST NOT alias the input list[T].
             meet: Callable[[T, T], T], # Correctness is not guaranteed if T meet treats T as a mutable object.
             forward: bool,
             start: list[T],
             top: T
            ) -> dict[BasicBlock, list[T]]:
    assert len(start) == len(function.basic_blocks), "Starting dataflow state (entry in forward, exit in backward) must have one element for each basic block in the function."
    num_blocks = len(start)
    top_vector = [top] * num_blocks

    ordering = postorder_traversal(function.entry_block, [], set())
    if forward:
        ordering.reverse() # Use reverse postorder for forward analyses.
    worklist = Queue() # Better data structure than a list for a queue.
    for item in ordering:
        worklist.put(item)

    in_states: dict[BasicBlock, list[T]] = {} # represents the INPUT states to each block (if forward) or the OUTPUT states (if backwards)
    out_states: dict[BasicBlock, list[T]] = {} # represents the OUTPUT states to each block (if forward) or the INPUT states (if backwards)
    
    for basic_block in function.basic_blocks:
        if is_starting_block(basic_block, function, forward):
            out_states[basic_block] = transfer_fn(start, basic_block)
        else:
            out_states[basic_block] = transfer_fn(top_vector, basic_block)
    
    # Begin worklist loop
    while not worklist.empty():
        current = worklist.get()

        if is_starting_block(current, function, forward):
            block_in = start
        else:
            block_in = top_vector
            for predecessor in predecessors(current, forward):
                block_in = vector_meet(block_in, out_states[predecessor], meet) # Generates a new list.
        
        in_states[current] = block_in
        block_out = transfer_fn(block_in, current)

        if block_out != out_states[current]:
            for successor in successors(current, forward):
                worklist.put(successor)
        
        out_states[current] = block_out
    
    return in_states


#### Helper methods for dataflow analysis   
def is_starting_block(basic_block: BasicBlock, function: Function, forward: bool):
    if forward:
        return id(basic_block) == id(function.entry_block)
    else:
        return len(basic_block.successors) == 0

def vector_meet(l: list[T], r: list[T], meet: Callable[[T, T], T]):
    output = []
    for l_i, r_i, in zip(l, r):
        output.append(meet(l_i, r_i))
    return output

def successors(basic_block: BasicBlock, forward: bool) -> Iterable[BasicBlock]:
    if forward:
        return basic_block.successors
    else:
        return basic_block.predecessors

def predecessors(basic_block: BasicBlock, forward: bool) -> Iterable[BasicBlock]:
    if forward:
        return basic_block.predecessors
    else:
        return basic_block.successors

def postorder_traversal(basic_block: BasicBlock, ordering: list[BasicBlock], encountered: set[BasicBlock]):
    # Ensure we don't record the same block twice and don't get stuck in a loop.
    if basic_block in encountered:
        return ordering
    encountered.add(basic_block)

    for successor in reversed(basic_block.successors):
        postorder_traversal(successor, ordering, encountered)
    
    ordering.append(basic_block)
    return ordering # The algorithm builds the postorder list via pass-by-reference, but returns the list here for the convenience of the original caller.




class Dominance:
    # Convention: True -> dominates, False -> does not dominate.
    def __init__(self, function: Function, forward=True):
        bb2idx = {}
        for i, basic_block in enumerate(function.basic_blocks):
            bb2idx[basic_block] = i
        
        def transfer(in_state: list[bool], bb: BasicBlock) -> list[bool]:
            out_state = in_state.copy()
            out_state[bb2idx[bb]] = True
            return out_state

        def meet(l: bool, r: bool) -> bool:
            return l and r
        
        self.strict_dominance_info: dict[BasicBlock, list[bool]] = dataflow(
            function, transfer, meet, forward, [False] * len(function.basic_blocks), True
        )
        self.bb2idx = bb2idx
        self.function = function
        self.forward = forward
    
    def strictly_dominates(self, x: BasicBlock, y: BasicBlock) -> bool:
        """Returns true if x sdom y.
        Precondition: x and y are basic blocks in the function that correspond to this Dominance instance.
        """
        assert x in self.strict_dominance_info, "BasicBlock x must be in the function that corresponds to this Dominance instance!"
        assert y in self.strict_dominance_info, "BasicBlock y must be in the function that corresponds to this Dominance instance!"

        idx = self.bb2idx[x]
        return self.strict_dominance_info[y][idx]

    
    def dominates(self, x: BasicBlock, y: BasicBlock) -> bool:
        """returns true if x dom y
        """
        if id(x) == id(y):
            return True
        return self.strictly_dominates(x, y)
    
    def dominance_frontier(self, x: BasicBlock) -> list[BasicBlock]:
        """Compute the dominance frontier of x
        """
        frontier = []
        for y in self.function.basic_blocks:
            if not self.strictly_dominates(x, y) and any(map(lambda pred_y: self.dominates(x, pred_y), predecessors(y, self.forward))):
                frontier.append(y)
        
        return frontier
    

class DominatorTree:
    class Node:
        def __init__(self, block: BasicBlock):
            self.block: BasicBlock = block
            self.children: set[DominatorTree.Node] = set()
        
        def __repr__(self):
            children_string = ", ".join([str(c.block.id) for c in self.children])
            return f"DominatorTree.Node({self.block.id}; children = {children_string})"
        
        def __hash__(self):
            return id(self.block)
        
        def __eq__(self, other):
            if not isinstance(other, DominatorTree.Node):
                return False
            return id(self.block) == id(other.block)
        
        def __iter__(self):
            """Iterate over the children of the DominatorTree Node.
            
            Ensures a consistent iteration order, unlike iterating directly from the set.
            """
            children = list(self.children)
            children.sort(key=lambda x: x.block.id)
            for child in children:
                yield child
    
    def __init__(self, function: Function, dominance: Dominance):
        self.function = function
        self.dominance = dominance
        self.node_count = 0
        self.visited = set()
        self.tree, undominated = self.make_tree(function.entry_block)
        assert len(undominated) == 0, "All blocks should be dominated by the entry node."
        del self.visited
        

    def make_tree(self, block: BasicBlock):
        # TODO: Make this work forward or backward (currently works for just forward).
        node = DominatorTree.Node(block)
        self.node_count += 1
        undominated = set() # This block does not dominate these basic blocks, but perhaps its parent might.
        for successor in block.successors:
            if successor not in self.visited:
                self.visited.add(successor)
                child_node, deeper_undominated = self.make_tree(successor)

                if self.dominance.dominates(block, successor):
                    node.children.add(child_node)
                    for subnode in deeper_undominated:
                        if self.dominance.dominates(block, subnode.block):
                            node.children.add(subnode)
                        else:
                            undominated.add(subnode)
                else:
                    undominated.add(child_node)
                    undominated.update(deeper_undominated)
        
        return node, undominated
    
    def __repr__(self):
        out = ""
        def build_repr(node):
            nonlocal out
            out += repr(node) + "\n"
            for child in node:
                build_repr(child)
        build_repr(self.tree)
        return out


###### Conversion to SSA Form ######
SSAProxyOperand = VarInstruction | Constant | Parameter | GlobalVariable
class PhiNodeProxy:
    def __init__(self, variable: Variable, out_repr: Optional[str] = None, differentiator: str | None = None):
        self.variable = variable
        self.out_repr = out_repr
        self.definitions: set[SSAProxyOperand] = set()
        self.deflist: list[SSAProxyOperand] = []
        self.ref_count = 0 # Records how many times this PhiNodeProxy is used.
        self.created = False
        self.ssa_node = SSAInstruction(Phi(variable, differentiator), []) # Will initialize these when possible.
        self.ssa_node.storage = variable # Normally would be done via the var_instruction parameter but ssa instructions have none.
    
    def __len__(self):
        return len(self.definitions)
    
    def __hash__(self) -> int:
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    def add_var_definition(self, var_instruction: SSAProxyOperand):
        if var_instruction not in self.definitions:
            self.deflist.append(var_instruction)
            self.definitions.add(var_instruction)
    
    def createPhiNode(self, var2ssa: dict[VarInstruction, SSAInstruction]):
        if self.created:
            return self.ssa_node

        phi_operands = []
        for definition in self.deflist:
            if isinstance(definition, VarInstruction):
                assert definition in var2ssa
                phi_operands.append(var2ssa[definition])
            else:
                phi_operands.append(definition) # definition is a parameter, global variable, constant, etc.

        self.created = True
        self.ssa_node.out_repr = self.out_repr
        self.ssa_node.operands = phi_operands
        return self.ssa_node
    
    def __repr__(self):
        argstrings = []
        for arg in self.deflist:
            if isinstance(arg, PhiNodeProxy):
                argstrings.append(f"{arg.variable} = PhiNodeProxy(...)") # prevent infinate loop in printing
            else:
                argstrings.append(repr(arg))
        argstring = ", ".join(argstrings)
        return f"{self.variable} = PhiNodeProxy[{self.ref_count}]({argstring})"

###############################################
### Main function for the conversion to SSA ###
###############################################
def convert_to_ssa(function: Function[VarInstruction], in_place=True, phi_differentiator: str | None = None) -> Function[SSAInstruction]:
    """Convert a function to single-static-assignment form.

    :param function: the function to convert
    :param in_place: whether or not to modify the function or return a new function with the original unchanged. Currently,
    only in_place=True is supported because tree-sitter AST-nodes can't be copied. To call the function with in_place=False,
    first set all ast_node instruction fields in the function to None.
    """
    if not in_place:
        function = copy.deepcopy(function)
    dominance = Dominance(function)

    # Information for placing phi nodes.
    orig: dict[BasicBlock, dict[Variable, VarInstruction]] = {}
    defsites: dict[Variable, set[BasicBlock]] = {}

    # Find all of the variables and global variables in the function. Track global variables separately; they
    # are handled differently during initialization
    variables: list[Variable] = [] # includes parameters.
    global_variables: list[GlobalVariable] = []
    for basic_block in function.basic_blocks:
        for instruction in basic_block:
            if isinstance(instruction.result, GlobalVariable):
                global_variables.append(instruction.result)
            elif instruction.result is not None:
                variables.append(instruction.result)
            for operand in instruction.operands:
                if isinstance(operand, GlobalVariable):
                    global_variables.append(operand)
                elif isinstance(operand, Variable):
                    variables.append(operand)

    orig[function.entry_block] = {}
    # Define variables at the top of the entry block. For parameters and global variables,
    # this represents them being defined at the start of the function. For local variables,
    # this represents an implicit initialization at the start of the function to the special
    # <uninitialized> value. If a variable is not initialized along a path before it is used,
    # this uninitialized value will be incorporated into the phi node.
    for variable in itertools.chain(variables, global_variables):
        # Temporaries are generated as part of expressions and thus should always dominate their uses.
        # This makes the output representations significantly better, because it prevents out_repr
        # numbers from being allocated to PhiNodeProxies that don't need them.
        if variable.is_temporary:
            continue
        if variable in orig[function.entry_block]:
            assert variable in defsites
        else:
            assert variable not in defsites
            if isinstance(variable, Parameter) or isinstance(variable, GlobalVariable):
                orig[function.entry_block][variable] = variable
            else:
                orig[function.entry_block][variable] = Uninitialized(variable)
            defsites[variable] = {function.entry_block}

    # Collect the definitions of each variable in each basic block, indexed both by basic block and by variable.
    # Store only the last definition of each variable in each basid block (done by overwriting each successive 
    # entry using a python dictionary indexed by variable).
    for basic_block in function.basic_blocks:
        if basic_block not in orig:
            orig[basic_block] = {}
        for instruction in basic_block:
            assert isinstance(instruction, VarInstruction)
            if instruction.result is not None:
                orig[basic_block][instruction.result] = instruction
                if instruction.result not in defsites:
                    defsites[instruction.result] = set()
                defsites[instruction.result].add(basic_block)

    # If necessary, add a unique differentiator to the phi nodes so that multiple uses of the same phi-node get different 
    # loop-phi variables. Unfortunately, we cannot use the SSA unique name (%0 etc.) because we create the Phi operations
    # before assigning the SSA differentiators. We could also use the memory address of the phi node, but this creates
    # cleaner output.
    if phi_differentiator:
        phi_differentiator = "_" + phi_differentiator # displays a bit better this way
    name_differentiators: dict[str, int] = {}
    def get_differentiator(variable: Variable) -> str | None:
        if variable.name in name_differentiators:
            name_differentiators[variable.name] = num = name_differentiators[variable.name] + 1
            return phi_differentiator + str(num) if phi_differentiator else f"_{num}"
        else:
            name_differentiators[variable.name] = 1 # means that the first phi will have no number and the second will have the number 2
            return phi_differentiator
    
    ### Place the phi nodes
    phi_nodes_to_add: dict[Variable, dict[BasicBlock, PhiNodeProxy]] = {}
    for variable, var_defsites in defsites.items():
        # It is unsafe to modify a data structure while iterating over it.
        # Thus we put the items of var_defsites in a queue for safe iteration.
        worklist: Queue[BasicBlock] = Queue()
        for site in var_defsites:
            worklist.put(site)

        # Stores where the phi nodes should be for a given variable (along with a precursor object of the phi node itself.)
        # This is stored in phi_nodes_to_add.
        var_phi_locations: dict[BasicBlock, PhiNodeProxy] = {}
        
        while not worklist.empty():
            current: BasicBlock = worklist.get()
            for frontier_block in dominance.dominance_frontier(current):
                if frontier_block not in var_phi_locations:
                    phi_node_proxy = PhiNodeProxy(variable, differentiator=get_differentiator(variable))
                    var_phi_locations[frontier_block] = phi_node_proxy
                    if variable not in orig[frontier_block]:
                        worklist.put(frontier_block)
                        # Because this block had no other definitions of this variable (variable not in orig[frontier_block]), 
                        # this is now the downward exposed definition of this variable in this block.
                        orig[frontier_block][variable] = phi_node_proxy
        
        phi_nodes_to_add[variable] = var_phi_locations

    # Determine what definitions need to go in each Phi node.
    for variable, phi_nodes_by_block in phi_nodes_to_add.items():
        for basic_block, phi_node in phi_nodes_by_block.items():
            # explore the predecessors of this block and collect definitions that reach this point.
            explored = set() # keep track of where we've visited to avoid repeatedly following a loop in the CFG.
            exploring: Queue[BasicBlock] = Queue()
            for predecessor in basic_block.predecessors:
                exploring.put(predecessor)
            
            while not exploring.empty():
                current_block = exploring.get()
                explored.add(current_block)
                if variable in orig[current_block]: # We encountered a definition of this variable in current_block.
                    # Add this definition to the phi node.
                    phi_node.add_var_definition(orig[current_block][variable])
                else: # We didn't encounter a definition here
                    for predecessor in current_block.predecessors:
                        if predecessor not in explored:
                            exploring.put(predecessor)

    # For each variable, stores where it was last defined. Used for assigning arguments when converting to SSA.
    var2def: dict[Variable, list[SSAOperand]] = {}

    for var in variables:
        var2def[var] = [Uninitialized(var)] # This list is used as a stack
    for parameter in function.parameters:
        var2def[parameter] = [parameter]
    for global_variable in global_variables:
        var2def[global_variable] = [global_variable]

    # Keep track of the var instruction from which each ssa instruction was generated.
    var2ssa: dict[VarInstruction, SSAInstruction] = {}
    
    # To make SSAInstructions' printed representations more readable.
    result_idx = 0
    def next_repr():
        nonlocal result_idx
        out_repr = f"%{result_idx}"
        result_idx += 1
        return out_repr

    def rename(dtree_node: DominatorTree.Node, var2value: dict[Variable, list[SSAOperand]]):
        """Now that phi-nodes have been inserted, "rename" variables such that each definition is unique
        and each definition dominates its uses. Because each definition is unique, we can represent each argument
        as an IR object reference to the operation that computes it.
        """
        nonlocal dominance
        nonlocal phi_nodes_to_add
        nonlocal var2ssa
        new_instructions: list[SSAInstruction] = []

        # Ensure that phi nodes at the start of this block can be used by its other instructions as arguments.
        for variable, phi_nodes_by_block in phi_nodes_to_add.items():
            if dtree_node.block in phi_nodes_by_block:
                phi_node_proxy = phi_nodes_by_block[dtree_node.block]
                if len(phi_node_proxy) > 1: # Phi nodes with one argument are irrelevant.
                    assert phi_node_proxy.out_repr is None
                    phi_node_proxy.out_repr = next_repr()
                    var2value[variable].append(phi_node_proxy)

        # Convert each instruction to SSA form, and do associated bookkeeping.
        for var_instruction in dtree_node.block:
            out_repr = None if var_instruction.result is None else next_repr()
            ssa_instruction = instruction_var_to_ssa(var_instruction, var2value, out_repr) # instruction_var_to_ssa updates var2value.
            new_instructions.append(ssa_instruction)
            var2ssa[var_instruction] = ssa_instruction # Used to resolve phi nodes' arguments later.
        
        for child in dtree_node:
            rename(child, copy_var2value(var2value))
        
        dtree_node.block.instructions = new_instructions    

    rename(DominatorTree(function, dominance).tree, var2def)

    # Not all phi nodes are necessary. Phi nodes that are never used by any downstream
    # instructions are unnecessary and can be removed. (Phi nodes with only one argument)
    # are also unnecessary, but that isn't handled here). We use the ref_count attribute
    # of PhiNodeProxy to determine how many times this phi node is used by downstream operations.
    # Some phi nodes are only used by other phi nodes. In the worklist algorithm below, we
    # propagate usage information backwards down the chains of phi nodes to ensure that they all
    # have the correct ref_counts. Using a worklist algorithm allows us to do this without
    # concern for the order that we process the PhiNodeProxies in.
    ref_chain_worklist: Queue[PhiNodeProxy] = Queue()
    for phi_node_proxy in (phi for _, phi_nodes_by_block in phi_nodes_to_add.items() for _, phi in phi_nodes_by_block.items()):
        ref_chain_worklist.put(phi_node_proxy)
    ref_count_info_propagated: set[PhiNodeProxy] = set()
    while not ref_chain_worklist.empty():
        phi_node_proxy = ref_chain_worklist.get()
        # Prevent infinate loops through the graph
        if phi_node_proxy in ref_count_info_propagated:
            continue
        # Phi node proxies of lenth less than 1 will be discarded; we don't want to increase reference
        # counts based on these. Additionally, we only want to increase reference counts for this phi
        # node's arguments if this phi node is itself used.
        if len(phi_node_proxy) > 1 and phi_node_proxy.ref_count > 0:
            ref_count_info_propagated.add(phi_node_proxy)
            for phi_operand in phi_node_proxy.deflist:
                if isinstance(phi_operand, PhiNodeProxy):
                    phi_operand.ref_count += 1
                    # This phi node may have not been considered before because its ref count was not high
                    # enough. Now that it definitely is, add it back to the worklist to be considered again
                    # if necessary.
                    ref_chain_worklist.put(phi_operand)

    # Filter out the irrelevant phi nodes (those that are unused or those that have only
    # one argument) and organize them by basic block. Create the phi nodes. (Note: the phi
    # nodes have technically already been created, but without their arguments. This is done
    # so that instructions can point to the phi nodes in those instructions' arguments. Calling
    # createPhiNode(var2ssa) here converts that shell of a phi node to a full phi node with 
    # arguments)
    bb2phi: dict[BasicBlock, list[SSAInstruction]] = {}
    for variable, phi_nodes_by_block in phi_nodes_to_add.items():
        for basic_block, phi_node_proxy in phi_nodes_by_block.items():
            if len(phi_node_proxy) > 1 and phi_node_proxy.ref_count > 0:
                if basic_block not in bb2phi:
                    bb2phi[basic_block] = []
                created = phi_node_proxy.createPhiNode(var2ssa)
                for i in range(len(created.operands)):
                    if isinstance(operand_phi_proxy := created.operands[i], PhiNodeProxy):
                        # Ensure that this phi node will be converted later.
                        assert len(operand_phi_proxy) > 1 and operand_phi_proxy.ref_count > 0, \
                            f"Attempting to create an invalid phi node for {variable} at basic_block {basic_block.id}: {phi_node_proxy}"
                        created.operands[i] = operand_phi_proxy.createPhiNode(var2ssa)
                bb2phi[basic_block].append(created)
    
    # Now that the SSA has been built, add the relevant phi-nodes to their basic blocks.
    for basic_block in function.basic_blocks:
        if basic_block in bb2phi: # This will only be true if there's at least one phi node for that block
            basic_block.instructions = bb2phi[basic_block] + basic_block.instructions # type: ignore (due to modifying the function in-place.)

    # Conversion to SSA complete. Return the function.
    return function # type: ignore (due to modifying the function in-place.)



def copy_var2value(var2value: dict[Variable, list[SSAOperand]]):
    """This is a copy that's in between a shallow and deep copy.
    We want to copy the enclosing dictionary and the lists of the values
    but not the keys or the elements of the value lists.
    """
    new_var2value = {}
    for key, value in var2value.items():
        new_var2value[key] = value[:] # Whole-list list slice shallow-copies the whole list.

    return new_var2value


def instruction_var_to_ssa(var_instruction: VarInstruction, var2value: dict[Variable, list[SSAOperand]], out_repr: str | None):
    ssa_operands = []
    for var_op in var_instruction.operands:
        ssa_operands.append(get_ssa_value(var_op, var2value))
    if isinstance(var_instruction.op, FunctionCall) and isinstance(var_instruction.op.fname, Variable):
        fname = get_ssa_value(var_instruction.op.fname, var2value)
        assert isinstance(fname, (Parameter, GlobalVariable, SSAInstruction)), f"Invalid SSA-representation for a function name."
        operation = FunctionCall(fname, var_instruction.op.ftype)
    else:
        operation = var_instruction.op
    ssa_instruction = SSAInstruction(operation, ssa_operands, out_repr=out_repr, var_instruction=var_instruction)
    
    # Update the current value of the variable.
    if var_instruction.result is not None:
        var2value[var_instruction.result].append(ssa_instruction)
    return ssa_instruction

def get_ssa_value(var_operand: VarOperand, var2value: dict[Variable, list[SSAOperand]]):
    # Could be a constant, field name, or type for a typecast
    if not isinstance(var_operand, Variable):
        return var_operand
    if (isinstance(var_operand, Parameter) or isinstance(var_operand, GlobalVariable)) and not var_operand in var2value:
        return var_operand
    value = var2value[var_operand][-1]
    if isinstance(value, PhiNodeProxy):
        value.ref_count += 1
        return value.ssa_node
    return value

def copy_propagation(function: Function):
    """precondition: function is in SSA form

    Modifies the function in place, eliminating all copy operations, replacing them with the copy op's argument.
    """

    val2use: dict[SSAInstruction, list[SSAInstruction]] = {} # track where each copy op is used.
    copy_ops: dict[SSAInstruction, BasicBlock] = {} # track where each copy op is located so we can easily delete it from that block later.

    # Find all copy ops. Find where each instruction is used.
    for basic_block in function.basic_blocks:
        for instruction in basic_block:
            for operand in instruction.operands:
                if isinstance(operand, SSAInstruction) and operand.op == COPY_OP: # could also be constants, parameters, etc.
                    if operand not in val2use:
                        val2use[operand] = []
                    val2use[operand].append(instruction)
            
            if instruction.op == COPY_OP:
                assert instruction not in copy_ops
                copy_ops[instruction] = basic_block
                if instruction not in val2use:
                    val2use[instruction] = []

    for copy_op, defblock in copy_ops.items():
        assert len(copy_op.operands) == 1
        for use in val2use[copy_op]:
            # use is an SSAInstruction
            for i in range(len(use.operands)):
                if use.operands[i] == copy_op:
                    use.operands[i] = copy_op.operands[0]
        
        defblock.instructions.remove(copy_op)

### Control dependence
def control_dependence(function: Function[InsT]) -> dict[BasicBlock[InsT], list[BasicBlock[InsT]]]:
    dependence: dict[BasicBlock, list[BasicBlock]] = {}
    dominance = Dominance(function, forward=False)
    for basic_block in function.basic_blocks:
        dependence[basic_block] = dominance.dominance_frontier(basic_block)
    return dependence

def instruction_level_strict_control_dependence(function: Function[InsT]) -> dict[InsT, tuple[InsT, ...]]:
    dependent = {}
    for block, block_dependencies in control_dependence(function).items():
        if block in block_dependencies:
            block_dependencies.remove(block)
        # All instructions in the same block share the same dependencies.
        instruction_dependencies = tuple(b.instructions[-1] for b in block_dependencies)
        assert all(isinstance(d.op, (LoopOp, If)) for d in instruction_dependencies), f"A control dependency must end in a branching instruction."
        for dependent_instruction in block:
            dependent[dependent_instruction] = instruction_dependencies
    return dependent

def control_equivalence_classes(function: Function) -> dict[tuple[BasicBlock,...], list[BasicBlock]]:
    classes: dict[tuple[BasicBlock,...], list[BasicBlock]] = {}
    for basic_block, dependencies in control_dependence(function).items():
        # Sort to ensure that the dependencies are in a consistent order. What that order is doesn't matter too much so long as it is consistent.
        # A tuple is requried because it is immutable and can therefore is hashable and suitable for use as a key in a dictionary.
        immutable_dependencies = tuple(sorted(dependencies, key=lambda x: x.id))
        if immutable_dependencies not in classes:
            classes[immutable_dependencies] = []
        classes[immutable_dependencies].append(basic_block)
    # Ensure a consistent output ordering for easier testing and debugging.
    for _, equivalent in classes.items():
        equivalent.sort(key=lambda x: x.id)
    return classes

### Loops
class Loop(Generic[InsT]):
    def __init__(self, head: BasicBlock[InsT], body: set[BasicBlock[InsT]], back_edge: tuple[BasicBlock[InsT], BasicBlock[InsT]]):
        self.head = head
        self.body = body
        self.back_edge = back_edge
        self.loop_branch, self.exit_blocks = self._find_exit_blocks()

    def _find_exit_blocks(self) -> tuple[BasicBlock[InsT], dict[BasicBlock[InsT], bool]]:
        loop_branch: BasicBlock[InsT] | None = None
        exit_blocks: dict[BasicBlock[InsT], bool] = {}

        worklist: deque[BasicBlock[InsT]] = deque()
        worklist.append(self.head)
        explored: set[BasicBlock[InsT]] = set()
        explored.add(self.head)
        while worklist:
            current = worklist.pop()
            if len(current.instructions) > 0 and isinstance(branch := current.instructions[-1].op, (If, LoopOp)):
                assert len(current.successors) == 2, f"A branch instruction must have exactly two successors"
                true_successor, false_successor = current.successors # faultless invariant: the true block comes first on all branch instructions.
                assert true_successor in self.body or false_successor in self.body, f"Invariant violation: all successors of a block in a loop exit the loop."
                true_exit = true_successor not in self.body and false_successor in self.body
                false_exit = true_successor in self.body and false_successor not in self.body

                if true_exit or false_exit:
                    if isinstance(branch, LoopOp):
                        assert loop_branch is None, f"{loop_branch.id} is already the loop branch; conflicts with {current.id}"
                        loop_branch = current
                    exit_blocks[current] = true_exit
            for successor in current.successors:
                if successor in self.body and successor not in explored:
                    worklist.append(successor)
                    explored.add(successor)
        
        assert loop_branch is not None, "Unable to find loop branch instruction for loop of back edge {}"
        return loop_branch, exit_blocks
            

def find_loops(function: Function[InsT]) -> list[Loop[InsT]]:
    """Find all loops in the given function.
    """
    dominance = Dominance(function)
    back_edges: list[tuple[BasicBlock[InsT], BasicBlock[InsT]]] = []
    visited: set[BasicBlock[InsT]] = set()

    def find_back_edges(block: BasicBlock):
        if block in visited:
            return
        visited.add(block)
        for successor in block.successors:
            if successor in visited and dominance.dominates(successor, block):
                # back edge found
                back_edges.append((block, successor))
            else:
                find_back_edges(successor)
    
    find_back_edges(function.entry_block)

    # Find a loop for each back edge.
    def find_loop(current: BasicBlock, head: BasicBlock, contents: set[BasicBlock]):
        if current == head:
            return
        contents.add(current)

        for predecessor in current.predecessors:
            if predecessor not in contents:
                find_loop(predecessor, head, contents)

    loops = []
    for back_edge in back_edges:
        tail, head = back_edge
        contents = set()
        find_loop(tail, head, contents)
        contents.add(head)
        loops.append(Loop(
            head=head,
            body=contents,
            back_edge=back_edge
        ))
    
    return loops

def init_loop_phi_base_cases(fn: Function[SSAInstruction]) -> int:
    """Updates, in place, the loop_base_case attributes of Phi operations. Returns the number of loop-phis identified."""
    loops = find_loops(fn)

    # Combine natural loops which share a head.
    merged: dict[BasicBlock[SSAInstruction], set[SSAInstruction]] = {}
    for loop in loops:
        if loop.head not in merged:
            merged[loop.head] = set()
        merged[loop.head].update(ins for bb in loop.body for ins in bb)

    loop_phi_count = 0
    for head, body in merged.items():
        for instruction in head:
            if isinstance(instruction.op, Phi):
                base_cases = [i for i, operand in enumerate(instruction.operands) if operand not in body]
                if len(base_cases) < len(instruction.operands): # otherwise this is not a loop-phi and we do nothing.
                    assert len(base_cases) == 1, f"Loop-phi instruction {instruction} has multiple base cases: {base_cases}"
                    instruction.op.loop_base_case = base_cases[0]
                    loop_phi_count += 1
    return loop_phi_count

