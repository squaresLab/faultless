"""Proves two symbolically executed functions equivalent, if possible.
"""

from collections import deque
from abc import ABC, abstractmethod

import z3

from .ir import *
from .analysis import find_loops, control_dependence, instruction_level_strict_control_dependence
from .interpreter import Execution, EquivalenceOptions, FunctionProgenitorSignature

class SynchronizationGraph:
    """This class represents a graph of the instructions which must match with the instructions
    in the other function during proof construction. The proof algorithm operates primarily on this 
    data structure by building a mapping between the nodes in two such graphs.
    """

    class Node():
        def __init__(self, instruction: SSAInstruction, progenitors: list[FunctionProgenitorSignature] | list[AddressableValue[Symbol]]):
            self.instruction = instruction
            self.progenitors = progenitors
            # The boolean indicates whether the edge includes/follows a back-edge.
            self.dependents: list[tuple["SynchronizationGraph.Node", bool]] = [] # Forward through the graph, following the flow of execution
            self.dependencies: list[tuple["SynchronizationGraph.Node", bool]] = [] # Backward through the graph, against the flow of execution

        @classmethod
        def make(cls, instruction: SSAInstruction, execution: Execution) -> "SynchronizationGraph.Node":
            match (instruction.op):
                case Phi(loop_base_case=loop_base_case, mediloop_value=mediloop_value, postloop_value=postloop_value):
                    assert loop_base_case is not None, f"Non-loop phi instruction {instruction} should not have a synchronization graph node."
                    return SynchronizationGraph.Phi(instruction, [mediloop_value, postloop_value])
                case FunctionCall():
                    symbols: list[FunctionProgenitorSignature] = [execution.call_vars[(instruction, True)]]
                    if (instruction, False) in execution.call_vars:
                        symbols.append(execution.call_vars[(instruction, False)])
                    return SynchronizationGraph.Call(instruction, symbols)
                case _:
                    raise ValueError(f"Invalid instruction type for a synchronization graph node: {instruction.op} ({instruction})")
        
        def __repr__(self):
            indent = "" if len(self.dependents) == 0 else "\n    "
            return f"{self.__class__.__name__}({self.instruction}){indent}" + "\n    ".join(
                f"{target.__class__.__name__}({target.instruction}, {follows_back_edge})" 
                for target, follows_back_edge in self.dependents
            )

    class Phi(Node):
        def __init__(self, instruction: SSAInstruction, progenitors: list[AddressableValue[Symbol]]):
            assert isinstance(instruction.op, Phi) and instruction.op.loop_base_case is not None # having a base case defined means it's a loop-phi
            super().__init__(instruction, progenitors)
    
    class Call(Node):
        def __init__(self, instruction: SSAInstruction, progenitors: list[FunctionProgenitorSignature]):
            assert isinstance(instruction.op, FunctionCall)
            super().__init__(instruction, progenitors)

    def __init__(self, adjacency_list: list[Node]):
        self.adjacency_list = adjacency_list
        
        # Compute the starting nodes from the adjacency list. A starting node has no dependencies except back edges.
        self.starting_nodes: list[SynchronizationGraph.Node] = [
            node for node in adjacency_list
            if sum(not follows_back_edge for _, follows_back_edge in node.dependencies) == 0
        ]
        assert len(self.adjacency_list) == 0 or len(self.starting_nodes) > 0, f"Nonempty synchronization graph does not have any starting nodes:\n{self}"

    def __repr__(self):
        start = "[" + ", ".join(f"{n.__class__.__name__}({n.instruction})" for n in self.starting_nodes) + "]"
        return f"SynchronizationGraph{start}:\n  " + "\n  ".join(str(node) for node in self.adjacency_list)


    @classmethod
    def build(cls, execution: Execution) -> "SynchronizationGraph":
        sync_nodes: dict[SSAInstruction, SynchronizationGraph.Node] = {}
        
        # Info for determining whether a given edge in the synchronization graph contains a back edge.
        ins2block: dict[SSAInstruction, BasicBlock] = {ins: bb for bb in execution.fn for ins in bb}
        # Exclude the head because there can be dependencies within the head that are not back edges.
        # For instance, consider "while (foo(bar(x))) { /* body */ }"
        loop_head2body: dict[BasicBlock, set[BasicBlock]] = {head: body - {head} for head, body in execution.merged_loops.items()}
        # Info for following control dependencies, since this is not provided directly in the instructions as dataflow edges are.
        control_dependencies = instruction_level_strict_control_dependence(execution.fn)
        # Info for determining if a dataflow dependency within a loop head is a back edge.
        insheadindices = {ins: i for head in loop_head2body for i, ins in enumerate(head)}

        # Identify the synchronization nodes.
        for block in execution.fn:
            for instruction in block:
                if isinstance(instruction.op, FunctionCall) or (isinstance(instruction.op, Phi) and instruction.op.loop_base_case is not None):
                    sync_nodes[instruction] = SynchronizationGraph.Node.make(instruction, execution)
        
        # For each synchronization instruction, search backwards through the use-def chains 
        # and then record and edge from any synchronization node found to itself. (SSA dataflow 
        # pointers in the operands are uses, pointing to definitions. This is the opposite 
        # direction of the way data actually flows through a program, so when recording edges
        # we record the opposite: from what we found to where we started.
        for sync_ins, sync_node in sync_nodes.items():
            worklist: deque[tuple[SSAInstruction, bool]] = deque()
            visited: set[SSAInstruction] = set()
            def add_successors(instruction: SSAInstruction, follows_back_edge: bool):
                for operand in itertools.chain(instruction.operands, control_dependencies[instruction]):
                    if isinstance(operand, SSAInstruction) and operand not in visited:
                        # A loop-carried dataflow dependency occurs when traversing from a loop phi
                        # at the head to one of its non-base-case incoming values from the loop body.
                        instruction_block = ins2block[instruction]
                        operand_block = ins2block[operand]
                        # Regular back edge going from inside the loop body to the head
                        is_back_edge = instruction_block in loop_head2body and operand_block in loop_head2body[instruction_block]
                        # Handle the case where there is a cyclic dependency within a loop head, for instance in while (i--) { ... }
                        if instruction_block == operand_block and instruction_block in loop_head2body:
                            assert not is_back_edge, f"is_back_edge is already true when "
                            # The = in the expression is important for the case when a phi instruction is self-referential. If a variable is modified
                            # on one path through the loop but not another, one of the loop-phi instruction's recursive arguments will be itself.
                            is_back_edge = insheadindices[operand] <= insheadindices[instruction]
                        worklist.append((operand, follows_back_edge or is_back_edge))
                        visited.add(operand)
            add_successors(sync_ins, False)

            while len(worklist) > 0:
                current, follows_back_edge = worklist.popleft()
                if current in sync_nodes:
                    # Add an edge from the node we found to the starting node for this search (this direction due to the backward search direction).
                    sync_nodes[current].dependents.append((sync_node, follows_back_edge))
                    # Add an edge from the current node to the node starting from 
                    sync_node.dependencies.append((sync_nodes[current], follows_back_edge))
                else:
                    add_successors(current, follows_back_edge)

        return SynchronizationGraph(list(sync_nodes.values()))

SGNode = SynchronizationGraph.Node

class SynchronizationProof:
    """A graph that tracks state during the proof process, represented as a mapping between nodes in two SynchronizationGraphs 
    and their truth state.

    The only sorts of propositions that we make in this graph are those that two nodes are equivalent to each other.
    """
    class Proposition(ABC):
        def __init__(self, lhs: SGNode, rhs: SGNode):
            """A "syntactic" expression of lhs==rhs without any assertion of truth value"""
            self.lhs = lhs
            self.rhs = rhs
            self.element_pair = (self.lhs, self.rhs)

        @abstractmethod
        def update_dependency_state(self, lemma: "SynchronizationProof.Lemma") -> bool:
            ...

        @abstractmethod
        def get_conjecture(self) -> "SynchronizationProof.Conjecture":
            ...

    class Lemma(Proposition):
        """A proven or assumed proposition that lhs == rhs.
        
        :param lhs: the left hand side of this lemma
        :param rhs: the right hand side of this lemma
        :param derivation: the conjecture used to prove this lemma. Contains the back-edges to prior nodes in the proof
        :param local_map: the new variables mapped together based on the completion of this lemma. Includes the global assumptions.
        :param global_assumptions: global variables that needed to be assumed equivalent in order to prove this lemma.
        """
        def __init__(self, lhs: SGNode, rhs: SGNode, derivation: "SynchronizationProof.Conjecture", local_map: VariableMap, global_assumptions: GlobalAssumptions):
            super().__init__(lhs, rhs)
            # Propositions proven based on (or derived from) this proposition.
            self.derivatives: list[SynchronizationProof.Conjecture] = []
            self.derivation = derivation # the Conjecture object contains info about the information used to prove this lemma.
            self.local_map = local_map
            self.global_assumptions = global_assumptions

        def update_dependency_state(self, lemma: "SynchronizationProof.Lemma") -> bool:
            # A lemma should have sufficient dependency state to be proven true. But it's possible to prove even more dependencies
            # equivalent than are strictly necessary; we track those additional ones as well. This is also important for back-dependencies,
            # which are assumed true by inductive assumption and then must be shown true; it allows us to determine if the induction was
            # successfully completed.
            return self.derivation.update_dependency_state(lemma)
        
        def get_conjecture(self):
            return self.derivation
        
        def __repr__(self):
            matrix = ("\n" + self.derivation._visualize_matrix(4)) if len(self.derivation.dependencies) > 0 else "    "
            return f"Lemma({self.lhs.instruction} == {self.rhs.instruction})" + matrix

    class Conjecture(Proposition):
        """An unproven expression that tracks evidence accumulated towards proof.
        
        In particular, it tracks which dependencies on the lhs are equivalent to which dependencies
        on the rhs. Each dependency is associated with a unique variable. The lhs and rhs expressions
        are defined in terms of those variables, so a necessary condition for showing lhs and rhs
        equivalent is showing each lhs dependency variable equivalent to a lhs variable, and vice versa.
        This is insufficient by itself, but means we can compare the corresponding expressions.
        """
        def __init__(self, lhs: SGNode, rhs: SGNode):
            super().__init__(lhs, rhs)

            # Matrix for tracking which lhs dependencies are equivalent to which rhs dependencies.
            # rows: lhs
            # cols: rhs
            self.dependencies: list[list[SynchronizationProof.Lemma | None]] = [
                [None for _ in range(len(rhs.dependencies))] 
                for _ in range(len(lhs.dependencies))
            ]

            # Row/Col index labels map dependency SynchronizationGraph nodes to row/column indices.
            #
            # The "has_counterpart" vectors signify if the dependency corresponding to this row/column 
            # has at least one corresponding dependency in the synchronization graph of the other function
            # OR if that dependency is a back-dependency. (Back dependencies are assumed equivalent by 
            # inductive assumption, then the validity of these assumptions is checked later.) In other 
            # words, it means that the corresponding row or column is non empty (not every entry is None.)
            # and/or that dependency is a back-dependency.
            self.row_index_labels, self.lhs_has_counterpart = SynchronizationProof.Conjecture._prepare_dependency_matrix_auxilliaries(lhs.dependencies)
            self.col_index_labels, self.rhs_has_counterpart = SynchronizationProof.Conjecture._prepare_dependency_matrix_auxilliaries(rhs.dependencies)
 
            self.dependency_mapping_plausible: bool = False
            self.check_plausibility()

        @staticmethod
        def _prepare_dependency_matrix_auxilliaries(dependencies) -> tuple[dict[SGNode, int], list[bool]]:
            has_counterpart = []
            index_labels = {}
            for i, (dependency, follows_back_edge) in enumerate(dependencies):
                index_labels[dependency] = i
                has_counterpart.append(follows_back_edge)
            return index_labels, has_counterpart
        
        def get_conjecture(self) -> "SynchronizationProof.Conjecture":
            return self
        
        def validate_mutually_equivalent_dependencies(self) -> bool:
            """Sets lhs_has_counterpart and rhs_has_counterpart to reflect the actual contents of the dependency table, overriding any
            inductive assuptions set up during this object's initialization. Returns whether or not the dependencies are mutually
            equivalent.
            """
            self.lhs_has_counterpart = [sum(r is not None for r in row) > 0 for row in self.dependencies]
            self.rhs_has_counterpart = [sum(c is not None for c in col) > 0 for col in zip(*self.dependencies)]
            self.check_plausibility()
            return self.dependency_mapping_plausible
        
        def check_plausibility(self):
            self.dependency_mapping_plausible = sum(self.lhs_has_counterpart) + sum(self.rhs_has_counterpart) \
                                             == len(self.lhs_has_counterpart) + len(self.rhs_has_counterpart)
        
        def update_dependency_state(self, lemma: "SynchronizationProof.Lemma") -> bool:
            l_idx = self.row_index_labels[lemma.lhs]
            r_idx = self.col_index_labels[lemma.rhs]

            self.dependencies[l_idx][r_idx] = lemma
            self.lhs_has_counterpart[l_idx] |= True
            self.rhs_has_counterpart[r_idx] |= True
            # Return true if there's a lemma in each row and each column, false otherwise.
            # This signifies that each synchonization variable in the left node's dependencies is equivalent to at least one synchonization variable
            # in the right node's dependencies.
            self.check_plausibility()
            return self.dependency_mapping_plausible
        
        def prove(self, local_map: VariableMap, global_assumptions: GlobalAssumptions) -> "SynchronizationProof.Lemma":
            # This assertion could technically be false in situations where the introduced synchonization variable is canceled out, e.g. bar(...) in
            #     x = foo(); y = x - x; bar(y); 
            # is equivalent to bar(0);, despite a different dependency structure.
            # However, this proof engine makes the (usually reasonable) assumption that this is not the case.
            assert self.dependency_mapping_plausible, f"Should not be able to prove that {self.lhs} == {self.rhs} due to implausible dependency mapping."
            return SynchronizationProof.Lemma(self.lhs, self.rhs, self, local_map, global_assumptions)
        
        def _visualize_matrix(self, indent: int = 0) -> str:
            pad = " " * indent
            rows = [node.instruction.out_repr or str(id(node.instruction)) for node in self.row_index_labels]
            cols = [node.instruction.out_repr or str(id(node.instruction)) for node in self.col_index_labels]
            cell_w = max([1, *(len(label) for label in cols)])
            row_w = max([0, *(len(label) for label in rows)])
            lines = [pad + " " * (row_w + 1) + " ".join(label.rjust(cell_w) for label in cols)]
            lines.extend(
                pad + row.rjust(row_w) + " " + " ".join(("*" if lemma is not None else ".").rjust(cell_w) for lemma in matrix_row)
                for row, matrix_row in zip(rows, self.dependencies)
            )
            return "\n".join(lines)
        
        def __repr__(self):
            matrix = ("\n" + self._visualize_matrix(4)) if len(self.dependencies) > 0 else ""
            return f"Conjecture({self.lhs.instruction} == {self.rhs.instruction})" + matrix
        

    def __init__(self, symbol_factory: MemorySymbolFactory, param_var_map: VariableMap):
        self.adjacency_list: dict[tuple[SGNode, SGNode], SynchronizationProof.Proposition] = {}
        self.param_var_map = param_var_map # Tracks which parameters are equivalent to eachother. The initial subset of var_map. Only contains entries that need to be here (because they have different inherent z3 representations).
        self.symbol_factory = symbol_factory # Tracks which fresh symbols are derived from which other symbols and tracks their indices to check that they are equivalent.

    def accumulate_symvars_from_progenitors(self, 
            lprogenitor: Sequence[Value | FunctionProgenitorSignature | None] | Value | FunctionProgenitorSignature | None, 
            rprogenitor: Sequence[Value | FunctionProgenitorSignature | None] | Value | FunctionProgenitorSignature | None, 
            context_map: VariableMap,
            accumulation_map: VariableMap,
            allow_uneven_lists: bool = False
        ):
        """Map progenitor symvars from across the synchronized executions to each other, as well as any symvars derived from them."""
        match (lprogenitor, rprogenitor):
            # Base case: a single progenitor value
            case (AddressableValue(base_address=lbase), AddressableValue(base_address=rbase)):
                assert isinstance(lbase, Symbol) and isinstance(rbase, Symbol)
                # Add the progenitor value if types allow.
                if accumulation_map.add_if_compatible(lbase, rbase): # Will return True on success.
                    # Then if we can add the progenitor value, we add all values derived from it
                    self.symbol_factory.derived_symbol_mapping(lbase, rbase, context_map, accumulation_map)
                    self.symbol_factory.search_cache(lbase, rbase, context_map, accumulation_map)
            # Recursive cases: multiple progenitor values, stored in some structured format.
            case (CompoundValue(offset_values=lvalues), CompoundValue(offset_values=rvalues)):
                offsets = sorted(set(lvalues).intersection(rvalues))
                for offset in offsets:
                    if offset in lvalues and offset in rvalues:
                        self.accumulate_symvars_from_progenitors(lvalues[offset], rvalues[offset], context_map, accumulation_map)
            case (FunctionProgenitorSignature(return_value=lret, arguments=largs), FunctionProgenitorSignature(return_value=rret, arguments=rargs)):
                self.accumulate_symvars_from_progenitors(lret, rret, context_map, accumulation_map)
                # We have a recursive case for sequences, so just use that for the lists. allow_uneven_lists for function calls where there are a 
                # mismatched number of arguments. Allowing this is okay even if that option is not set because this functionn is only called on lemmas, 
                # and when the option is not set functions with mismatched numbers of arguments won't be able to be proven lemmas.
                self.accumulate_symvars_from_progenitors(largs, rargs, context_map, accumulation_map, allow_uneven_lists=True)
            case ([*lprogenitors], [*rprogenitors]):
                if allow_uneven_lists or len(lprogenitors) == len(rprogenitors):
                    for l, r in zip(lprogenitors, rprogenitors):
                        self.accumulate_symvars_from_progenitors(l, r, context_map, accumulation_map)
            case _:
                pass

    def get_var_map_for(self, element_pair: tuple[SGNode, SGNode]) -> VariableMap:
        """For each lemma, return the corresponding pair of equivalent variables.
        If an element_pair is specified, return the var map only for the transitive dependencies of the variable.

        precondition: if element_pair is specified, it is in the adjacency list.
        """
        assert element_pair in self.adjacency_list, f"Can only get var_map for established propositions; {element_pair} has not yet been established."
        var_map = self.param_var_map.copy()
        visited: set[tuple[SGNode, SGNode]] = set()
        worklist: deque[SynchronizationProof.Lemma] = deque(
            dependency for row in self.adjacency_list[element_pair].get_conjecture().dependencies
            for dependency in row if dependency is not None
        )

        while worklist:
            dependency = worklist.popleft()
            if dependency.element_pair in visited:
                continue
            visited.add(dependency.element_pair)

            var_map.update(dependency.local_map)

            worklist.extend(
                transitive_dependency for row in dependency.get_conjecture().dependencies
                for transitive_dependency in row if transitive_dependency is not None
            )
        return var_map

    def is_proven(self, element_pair: tuple[SGNode, SGNode]):
        return element_pair in self.adjacency_list and isinstance(self.adjacency_list[element_pair], SynchronizationProof.Lemma)

    def mark_as_proven(self, proven_pair: tuple[SGNode, SGNode], proof_context: VariableMap, global_assumptions: GlobalAssumptions) -> list[tuple[SGNode, SGNode]]:
        assert proven_pair in self.adjacency_list, f"Cannot prove {proven_pair} because it has not been proposed in the proof."
        conjecture = self.adjacency_list[proven_pair]
        assert isinstance(conjecture, SynchronizationProof.Conjecture), f"Can only prove a conjecture but found {conjecture}"
        local_map = VariableMap()
        local_map.update(global_assumptions)
        self.accumulate_symvars_from_progenitors(conjecture.lhs.progenitors, conjecture.rhs.progenitors, proof_context, local_map)
        self.adjacency_list[proven_pair] = lemma = conjecture.prove(local_map, global_assumptions)

        deps_satisfied: list[tuple[SGNode, SGNode]] = []
        for ldep, _ in proven_pair[0].dependents:
            for rdep, _ in proven_pair[1].dependents:
                if type(ldep) == type(rdep):
                    element_pair = (ldep, rdep)
                    self.add_conjecture(element_pair) # adds if it doesn't already exist, otherwise, no-op.
                    downstream_conjecture = self.adjacency_list[element_pair].get_conjecture()
                    if downstream_conjecture.update_dependency_state(lemma):
                        deps_satisfied.append(element_pair)
                    lemma.derivatives.append(downstream_conjecture)
        return deps_satisfied

    def add_conjecture(self, element_pair: tuple[SGNode, SGNode]):
        """Adds an element pair to the graph as a conjecture if it is not already on the graph.
        """
        # It is normal to see the same pair added multiple times when a given node has multiple dependencies and each 
        # at least two pairs of those dependencies are proven equivalent. There's no need to re-add here.
        if element_pair not in self.adjacency_list:
            self.adjacency_list[element_pair] = SynchronizationProof.Conjecture(*element_pair)

    def revoke(self, element_pair: tuple[SGNode, SGNode]):
        """Downgrades the truth status of a given proposition (represented by an SGNode pair). If the proposition is a Lemma, it is reverted
        to a conjecture, and any downstream derivatives proved based on it are downgraded to conjectures as well, with the corresponding derivation
        state downgraded as well. If the proposition is a conjecture and has no evidence at all (i.e. no dependencies), then it is removed from the
        graph entirely.
        """
        assert element_pair in self.adjacency_list, f"{element_pair} is not in this SynchronizationProof."
        node = self.adjacency_list[element_pair]
        if isinstance(node, SynchronizationProof.Lemma):
            # Delete the node from the adjacency list before searching through its dependencies. This prevents processing 
            # the same node multiple times if multiple paths through the proof reach the same node (including through loops).
            # Conjectures are terminal nodes in the derivation chain so we don't have to worry about that happening with them.
            self.adjacency_list[element_pair] = node.derivation # downgrade to a conjecture.
            for derivative in node.derivatives:
                if derivative.element_pair in self.adjacency_list: # could have been removed on a prior traversal through the graph.
                    l_idx = derivative.row_index_labels[element_pair[0]]
                    r_idx = derivative.col_index_labels[element_pair[1]]

                    assert derivative.dependencies[l_idx][r_idx] == node, f"SynchronizationProof inconsistency: {element_pair} lists {derivative.element_pair} as a dependency but this lemma is not found in the dependency table."
                    derivative.dependencies[l_idx][r_idx] = None
                    if not derivative.validate_mutually_equivalent_dependencies() and self.is_proven(derivative.element_pair):
                        self.revoke(derivative.element_pair)
            node = node.derivation # then if necessary the conjecture can be deleted as well in the if statement below.
        if isinstance(node, SynchronizationProof.Conjecture):
            if all(dep is None for row in node.dependencies for dep in row):
                del self.adjacency_list[element_pair]

    def revoke_incorrect_global_assumptions(self, 
            lstack: Stack, rstack: Stack, 
            lprogenitor2var: dict[str, GlobalVariable], rprogenitor2var: dict[str, GlobalVariable],
            lderivation: dict[str, str], rderivation: dict[str, str]
        ):
        """Recursively remove any lemmas supported by a global assumption for global variables with non-equivalent values.
        This is done by seeing if the values stored in the global variables are equivalent at the end of the function (or if nothing
        was written to them). Global variables storing different values can't correspond to each other, and therefore we must 
        revoke any lemma proved based on such an erroneous assumption.

        lstack, rstack: the stack state at the function return
        lprogenitor2var, rprogenitor2var: maps each progenitor symbol name to the corresponding GlobalVariable object it was derived from.
            (The global variable represents the memory location where the data is stored and the symbol represents the initial value.)
        lderivation, rderivation: maps each derived symbol name to the corresponding progenitor symbol name.
        """
        # We'll be calling revoke() which modifies the adjacency list so we have to make a copy of the keys to avoid iterator issues.
        element_pairs: list[tuple[SGNode, SGNode]] = list(self.adjacency_list)
        lvalue_cache: dict[GlobalVariable, Value | None] = {}
        rvalue_cache: dict[GlobalVariable, Value | None] = {}

        def get_value(symbol_name: str, progenitors: dict[str, str], variables: dict[str, GlobalVariable], cache: dict[GlobalVariable, Value | None], stack: Stack):
            variable = variables[progenitors[symbol_name]]
            if variable in cache:
                return cache[variable]
            else:
                value = read_written_stack_contents(stack, variable)
                cache[variable] = value
                return value
 
        for element_pair in element_pairs:
            if element_pair not in self.adjacency_list:
                continue # it's been revoked, we can ignore it.
            lemma = self.adjacency_list[element_pair]
            if not isinstance(lemma, SynchronizationProof.Lemma):
                continue
            for left_symbol_name, right_symbol_name in lemma.global_assumptions.symbol_mapping():
                lvalue = get_value(left_symbol_name, lderivation, lprogenitor2var, lvalue_cache, lstack)
                rvalue = get_value(right_symbol_name, rderivation, rprogenitor2var, rvalue_cache, rstack)
                if lvalue is not None and rvalue is not None:
                    context = self.get_var_map_for(element_pair)
                    equivalent = equivalent_values(lvalue, rvalue, var_map=context, permissive_typing=True)
                else:
                    equivalent = lvalue is None and rvalue is None
                
                if not equivalent:
                    self.revoke(element_pair)


    def build_full_var_map(self):
        """Build a variable mapping containing the mappings of all proven lemmas."""
        var_map = self.param_var_map.copy()
        for proposition in self.adjacency_list.values():
            if isinstance(proposition, SynchronizationProof.Lemma):
                var_map.update(proposition.local_map)
        return var_map
    
    def equivalent_calls(self) -> dict[SSAInstruction, list[SSAInstruction]]:
        """For each call in both functions, return the calls in the other function which are equivalent to this call.
        Because callsites hash and compare by ID/memory location, we can store both in the same dictionary.
        """
        equivalent_calls: dict[SSAInstruction, list[SSAInstruction]] = {}
        for (left, right), proposition in self.adjacency_list.items():
            if isinstance(proposition, SynchronizationProof.Lemma):
                if isinstance(left, SynchronizationGraph.Call) or isinstance(right, SynchronizationGraph.Call):
                    assert isinstance(right, SynchronizationGraph.Call) and isinstance(right, SynchronizationGraph.Call)
                    if left.instruction in equivalent_calls:
                        equivalent_calls[left.instruction].append(right.instruction)
                    else:
                        equivalent_calls[left.instruction] = [right.instruction]
                    if right.instruction in equivalent_calls:
                        equivalent_calls[right.instruction].append(left.instruction)
                    else:
                        equivalent_calls[right.instruction] = [left.instruction]
        return equivalent_calls
    
    def __repr__(self):
        return "SynchronizationProof\n  " + "\n  ".join(repr(prop) for prop in self.adjacency_list.values())

def read_written_stack_contents(stack: Stack, variable: Variable, condition: z3.BoolRef | bool = True) -> Value | None:
    """Return the value on the stack for the provided variable, or None if nothing was written to the stack during the execution of the function."""
    if (stack.contains_exactly_initial_write(variable) if isinstance(variable, (Parameter, GlobalVariable)) else stack.address_space_is_empty(variable)):
        return None
    assert isinstance(variable.type, ObjectType)
    return stack.read(variable, Offset(z3_zero(Pointer(Void())).expr, condition, variable.type.get_size()), variable.type)
   
def align_subarguments(left: Value, right: Value, arguments: list[tuple[Value, Value]]) -> bool:
    if isinstance(left, CompoundValue) and isinstance(right, CompoundValue):
        if not isinstance(left.type, Struct) == isinstance(right.type, Struct):
            return False
        if left.offsets() != right.offsets():
            return False
        for (loffset, left_value), (roffset, right_value) in zip(left, right):
            assert loffset == roffset, f"Expected CompoundValues to return offsets in ascending order."
            if not align_subarguments(left_value, right_value, arguments):
                return False
    elif isinstance(left, CompoundValue) or isinstance(right, CompoundValue):
        return False
    else:
        arguments.append((left, right))
    return True

def equivalent_calls(lcall: tuple[str | Value, list[Value], Stack, Heap, bool | z3.BoolRef], 
                     rcall: tuple[str | Value, list[Value], Stack, Heap, bool | z3.BoolRef], 
                     context_map: VariableMap,
                     global_assumptions: GlobalAssumptions,
                     equivalence_options: EquivalenceOptions
                    ) -> bool:
    """Determine if two function calls are equivalent and return True if so, False otherwise.
    """
    lname, largs, lstack, lheap, lcond = lcall
    rname, rargs, rstack, rheap, rcond = rcall

    if not equivalence_options.ignore_extra_arguments and len(largs) != len(rargs):
        return False
    
    condition: z3.BoolRef = z3.And(lcond, rcond) # type: ignore
    solver = z3.Solver()
    
    if isinstance(lname, Value) and isinstance(rname, Value): # if: Two function pointers. Check to see if they are equivalent in the context of the function.
        if not equivalent_values(lname, rname, condition, context_map, global_assumptions, permissive_typing=True):
            return False
    elif isinstance(lname, Value) or isinstance(rname, Value): # elif: one argument is a function pointer, the other is not.
        return False
    elif equivalence_options.require_exact_function_names and lname != rname: # else: string i.e. non-function-pointer names. On
        return False
    
    ### Arguments must be positionally equivalent.
    arguments = []
    for larg, rarg in zip(largs, rargs):
        # TODO: incorporate the function signatures of the called functions if they are known and compatible.

        # A common pattern in C is to initialize variables by reference through calls to initializer functions like scanf.
        # When infer_stack_initializer_functions is set, we assume that functions to which the address of an uninitialized
        # stack variable is passed are initializer functions. Thus, the value of passed to the initializer functions are 
        # irrelevant, so we can ignore them. That is what this condition checks. Both functions under consideration for
        # equivalence must be initializers in order to be equivalent (otherwise one of the functions would care about the 
        # value of the argument), so we check both simultameously.
        if equivalence_options.infer_stack_initializer_functions and \
           isinstance(larg, AddressableValue) and isinstance(larg.base_address, Variable) and lstack.address_space_is_empty(larg.base_address) and \
           isinstance(rarg, AddressableValue) and isinstance(rarg.base_address, Variable) and rstack.address_space_is_empty(rarg.base_address):
            assert isinstance(larg.type, (Pointer, Array)) and isinstance(rarg.type, (Pointer, Array)) # Variable-based AddressableValues represent pointers to the memory on the stack.
            # Ensure that we're reading the same types into memory.
            match (larg.type, rarg.type):
                case (Pointer(target_type=ltgt), Pointer(target_type=rtgt)):
                    compatible = bool(resolve_to_compatible_z3_repr(ltgt, rtgt))
                case (Array(element_type=letype, nelements=llen), Array(element_type=retype, nelements=rlen)):
                    compatible = bool(resolve_to_compatible_z3_repr(letype, retype)) and llen == rlen
                case _:
                    compatible = False
            if compatible:
                continue # then these arguments are inferred as initialization arguments and no further action is required.
            else:
                return False

        # For pointers to values on the stack, we read the values at those pointers.
        # The values of the pointers themselves are symvars that reference the variable names of the pointers.
        # We don't have an alignment between stack variables, nor do we want to compute one.
        # What we actually care about when calling functions is to what those variables point.
        # This is different than for heap variables, where we must show both the pointers are equivalent
        # (relative to some other pointer source like arguments or allocating call returns) AND the contents 
        # of the memory at those locations equivalent.
        if isinstance(larg, AddressableValue) and isinstance(larg.base_address, Variable):
            assert isinstance(larg.type, ObjectType)
            # We use z3_zero(Pointer(Void())) because the z3_zero argument target type doesn't matter.
            larg = lstack.read(larg.base_address, Offset(z3_zero(Pointer(Void())).expr, lcond, larg.type.get_size()), larg.base_address.type)
        if isinstance(rarg, AddressableValue) and isinstance(rarg.base_address, Variable):
            assert isinstance(rarg.type, ObjectType)
            rarg = rstack.read(rarg.base_address, Offset(z3_zero(Pointer(Void())).expr, rcond, rarg.type.get_size()), rarg.base_address.type)
        
        # Common case which can be checked quickly without z3. The path conditions are irrelevant because string literals
        # are initialized in the immutable ROData section before the program begins running.
        if isinstance(larg, StringLiteral) and isinstance(rarg, StringLiteral):
            if larg == rarg:
                continue
            else:
                return False
        
        # Handles scalars as well as composite values. Base case is just a regular argument.
        if not align_subarguments(larg, rarg, arguments):
            return False
    
    
    for larg, rarg in arguments:
        if not equivalent_values(larg, rarg, condition, context_map, global_assumptions, True, solver):
            return False
    
    ### Compare the heap state of addressable arguments.
    # This condition is theoretically able to introduce unsoundness due to limitations with faultless'
    # addressing model. Addressability is not preserved in situations where two addressable values are added
    # or subtracted (like in "void foo(long ptr, long n) { bar(ptr + n);}"). Here, bar's argument could feasibly
    # be typecast to a pointer and dereferenced. Addressability is also not preserved under other operations
    # like multiplication, division, bit-shifts, etc. but it's very difficult to use those in code on memory
    # addresses in any meaningful way that doesn't introduce a segfault.
    checked: set[tuple[Symbol, Symbol]] = set()
    for larg, rarg in zip(largs, rargs):
        lbase = larg.base_address if isinstance(larg, AddressableValue) and isinstance(larg.base_address, Symbol) else None
        rbase = rarg.base_address if isinstance(rarg, AddressableValue) and isinstance(rarg.base_address, Symbol) else None
        if isinstance(lbase, Symbol) and isinstance(rbase, Symbol):
            if (lbase, rbase) not in checked: # No need to redo an expensive query if we don't have to
                if not equivalent_heaplets(lheap, rheap, larg, rarg, context_map, global_assumptions, equivalence_options): # type: ignore -- the type system does not recognize that larg is always an AddressableValue when lbase is a Symbol
                    return False
                checked.add((lbase, rbase))
        # Mismatched addressability. Returning false/nonequivalent here is a very conservative choice.
        elif type(lbase) != type(rbase):
            return False

    return True

def equivalent_heaplets(lheap: Heap, rheap: Heap, 
                        laddr: AddressableValue[Symbol], raddr: AddressableValue[Symbol],
                        var_map: VariableMap,
                        global_assumptions: GlobalAssumptions,
                        equivalence_options: EquivalenceOptions
                       ) -> bool:
    """Determine if the heaplets at laddr and raddr are equivalent at all addresses.
    """
    left_is_empty = lheap.address_space_is_empty(laddr.base_address)
    right_is_empty = rheap.address_space_is_empty(raddr.base_address)

    if left_is_empty != right_is_empty:
        return False # technically could still be equivalent if the path condition on the write is a contradiction, but that is unlikely and this is still sound.
    if left_is_empty and right_is_empty:
        return True   

    if isinstance(laddr.base_address.type, (Array, Struct)) or isinstance(raddr.base_address.type, (Array, Struct)):
        raise NotImplementedError(f"Support for Array or Struct variables in heapspace comparisons is not implemented.")
    if equivalence_options.memory_formatting_from_index is not None:
        canonical_base_address = (laddr, raddr)[equivalence_options.memory_formatting_from_index]
        if isinstance(canonical_base_address.type, Pointer):
            lread_t = rread_t = MemoryOperation.type_at_address(canonical_base_address)
        else:
            lread_t = rread_t = INTEGER
    else:
        if isinstance(laddr.base_address.type, Pointer) and isinstance(raddr.base_address.type, Pointer):
            lread_t = MemoryOperation.type_at_address(laddr)
            rread_t = MemoryOperation.type_at_address(raddr)
        elif isinstance(laddr.base_address.type, Pointer):
            lread_t = rread_t = MemoryOperation.type_at_address(laddr)
        elif isinstance(raddr.base_address.type, Pointer):
            lread_t = rread_t = MemoryOperation.type_at_address(raddr)
        else:
            lread_t = rread_t = INTEGER

    if isinstance(lread_t, (IncompleteStruct, IncompleteUnion)):
        lread_t = lread_t.full_definition
    if isinstance(rread_t, (IncompleteStruct, IncompleteUnion)):
        rread_t = rread_t.full_definition

    assert isinstance(lread_t, ObjectType) and isinstance(rread_t, ObjectType)
    lsize = lread_t.get_size()
    rsize = rread_t.get_size()
    element_var = z3repr((Pointer(Void()), "\\element"))
    # The multiplication read_size * element_var is important to constrain the read to be only at the memory slots that the variable  could occupy.
    # This is especially important to avoid "frame shift" type errors in reading. For instance, suppose that we have a pointer
    # to a struct with definition struct s { int i; float f; }; and we write to each element once (x->i = 3; x->f = 1.5;).
    # Then if we were to read from each index, when reading from index 4, we'd try to read the 1.5 into the i field and allocate 
    # a fresh variable for the f field. The former read is problematic, since it misrepresents the type of 1.5 by reading it into an int.
    # Additionally, with the way the memory model works, most of the memory slots are empty and it doesn't make sense to search over them.
    # Rather, it makes sense to search over only the valid indices that the item would be placed in if memory was treated as an array starting
    # at the base address. Structs will be unpacked and each field will be read in from the appropriate address.
    lmem = lheap.read(laddr.base_address, Offset(lsize * element_var, True, lsize), lread_t)
    rmem = rheap.read(raddr.base_address, Offset(rsize * element_var, True, rsize), rread_t)
    
    assert lheap.symbol_factory is rheap.symbol_factory, f"Symbol factory sholuld be shared across executions to ensure consistent derived variable naming."

    # TODO: look into this. May be able to use "global assumptions" as the local variable map.
    local_map = var_map
    if laddr.base_address != raddr.base_address:
        equivalent_fresh = lheap.symbol_factory.derived_symbol_mapping(laddr.base_address, raddr.base_address, var_map)
        if len(equivalent_fresh) > 0:
            local_map = var_map + equivalent_fresh # treat the input var_map as immutable
    
    # TODO: recursively search memory for nested data structures.

    return equivalent_values(lmem, rmem, None, local_map, global_assumptions, permissive_typing=True)

def dependent_condition(block: BasicBlock, control_dependencies: dict[BasicBlock, list[BasicBlock]], path_condition: PathCondition) -> z3.BoolRef:
    """Return only the components of the path condition that determine if this basic block is executed."""
    components: ComponentwisePathT = {}
    bb2ntg: dict[BasicBlock, NonTautologicalGroup] = {}
    for component in path_condition.components:
        if isinstance(component, NonTautologicalGroup):
            for bb in component.blocks:
                bb2ntg[bb] = component

    worklist: deque[BasicBlock] = deque(control_dependencies[block])
    visited: set[BasicBlock | NonTautologicalGroup] = set()
    while worklist:
        dependent = worklist.popleft()
        if dependent in bb2ntg:
            dependent = bb2ntg[dependent]
        if dependent in visited:
            continue
        visited.add(dependent)
        # If the dependent is not in the path, it could be a basic block that is downstream from the current block, for instance,
        # in a loop with a break statement. (The loop head is dependent on the if with the break and the if is dependent on the
        # loop head.) Faultless' models loop bodies like functions that are executed once, with the bodies of those functions checked
        # for equivalence. Therefore, downstream path conditions won't be in the path and aren't relevent under this execution model.
        if dependent in path_condition.components:
            components[dependent] = path_condition.components[dependent]
            if isinstance(dependent, NonTautologicalGroup):
                for bb in dependent.blocks:
                    worklist.extend(d for d in control_dependencies[bb])
            else:
                invariant_key = LoopInvariantKey(dependent)
                if invariant_key in path_condition.components:
                    components[invariant_key] = path_condition.components[invariant_key]
                worklist.extend(d for d in control_dependencies[dependent])
    return PathCondition(components).expr()

def loop_continuation_condition(loop_phi_instruction: SSAInstruction, execution: Execution, control_dependencies: dict[BasicBlock, list[BasicBlock]]) -> z3.BoolRef:
    """Return the path condition required for the loop to complete."""
    head = execution.ins2bb[loop_phi_instruction]
    assert head in execution.back_edge_conditions and head in execution.loop_head_entry_conditions, f"Expected {loop_phi_instruction} to be a loop-phi instruction but could not find path information."
    back_edge_components = execution.back_edge_conditions[head].components
    loop_base_components = execution.loop_head_entry_conditions[head].components
    assert all(lbc in back_edge_components for lbc in loop_base_components), f"All loop base conditions should be in the back edge condition as background components but found\n{execution.back_edge_conditions[head].expr()}\n  and\n{execution.loop_head_entry_conditions[head].expr()}"
    context_free_components: ComponentwisePathT = {c: cond for c, cond in back_edge_components.items() if c not in loop_base_components and not isinstance(c, LoopInvariantKey)}
    return dependent_condition(head, control_dependencies, PathCondition(context_free_components))

def parameter_symbols_by_location(arguments: list[AddressableValue[Symbol] | CompoundValue]) -> dict[str, AddressableValue[Symbol]]:
    name2symbol: dict[str, AddressableValue[Symbol]] = {}
    for argument in arguments:
        if isinstance(argument, CompoundValue):
            for value in argument.decompose().values():
                assert isinstance(value, AddressableValue) and isinstance(value.base_address, Symbol)
                name2symbol[value.base_address.name.split("param")[-1]] = value
        else:
            name2symbol[argument.base_address.name.split("param")[-1]] = argument
    return name2symbol


# could also consider making this a classmethod for a SynchronizationProof object
def synchronize(left: Execution, right: Execution, symbol_factory: MemorySymbolFactory, equivalence_options: EquivalenceOptions) -> SynchronizationProof:
    """Inductively builds a product program, proving pairs of relevant instructions at cut points equivalent.
    """
    left_graph = SynchronizationGraph.build(left)
    right_graph = SynchronizationGraph.build(right)

    # This info will be useful for limiting the path conditions for function calls to the
    # components of the path that the calls are dependent on
    lcontrol_deps = control_dependence(left.fn)
    rcontrol_deps = control_dependence(right.fn)

    # Initialize the var_map with equivalent parameters. While equivalent parameters already
    # have the same name, if they have different z3 sorts, the z3 solver considers them different
    # variables, thus undermining one of the key sources of premises for the proof. Thus, we 
    # manually align them if necessary.
    lparams = parameter_symbols_by_location(left.arguments)
    rparams = parameter_symbols_by_location(right.arguments)
    param_var_map = VariableMap()
    for symname, lparam in lparams.items():
        if symname in rparams:
            rparam = rparams[symname]
            if param_var_map.add_if_compatible(lparam.base_address, rparam.base_address):
                param_var_map.update(symbol_factory.derived_symbol_mapping(lparam.base_address, rparam.base_address, param_var_map))
    
    proof = SynchronizationProof(symbol_factory, param_var_map)
    worklist: deque[tuple[SGNode, SGNode]] = deque()
    for left_start in left_graph.starting_nodes:
        for right_start in right_graph.starting_nodes:
            if type(left_start) == type(right_start):
                current_pair = (left_start, right_start)
                worklist.append(current_pair)
                proof.add_conjecture(current_pair)

    # Track the pairs of phi-instruction nodes for which have shown the inductive hypothesis.
    # This is not tracked by any state in the proof object.
    induction_complete: set[tuple[SGNode, SGNode]] = set()
    
    # Main proof loop. Prove pairs of synchronization nodes equivalent.
    while len(worklist) > 0:
        left_node, right_node = current_pair = worklist.popleft()

        # Record assumptions, keeping them separate from the overall var_map. We only add these assumptions
        # to the var_map if the proposition is proven.
        global_assumptions = GlobalAssumptions(left.globals, right.globals, symbol_factory) # Represents new assumptions we make in proving this node equivalent.
        context_map = proof.get_var_map_for(current_pair) # Built from this node's proven dependencies. Treated as immutable background context.
        
        equivalent: bool = False
        if isinstance(left_node, SynchronizationGraph.Call) and isinstance(right_node, SynchronizationGraph.Call):
            if not proof.is_proven(current_pair):
                # Show control flow context equivalent
                lcond = dependent_condition(left.ins2bb[left_node.instruction], lcontrol_deps, left.call_conditions[left_node.instruction])
                rcond = dependent_condition(right.ins2bb[right_node.instruction], rcontrol_deps, right.call_conditions[right_node.instruction])
                # If control flow is different, there will be observable differences in side effects on some inputs (whether or not the call is executed).
                # Thus, matching control flow is a necessary condition and we can exit early if this is not true.
                if equivalent_expressions(lcond, rcond, var_map=context_map, global_assumptions=global_assumptions):
                    # All calls will exist on the first iteration. If the call is in a loop head, it might have a second instance of the call as well.
                    # First, we check to see that the calls either both have a second call instance or don't. If this is not true, then they don't align.
                    if ((left_node.instruction, False) in left.calls) == ((right_node.instruction, False) in right.calls):
                        # Next, we check that the first instances of the calls are the same
                        equivalent = equivalent_calls((*left.calls[(left_node.instruction, True)], lcond), (*right.calls[(right_node.instruction, True)], rcond), context_map, global_assumptions, equivalence_options)
                        if (left_node.instruction, False) in left.calls:  # If there are second call instances for both, then we check those equivalent as well.
                            equivalent = equivalent and equivalent_calls((*left.calls[(left_node.instruction, False)], lcond), (*right.calls[(right_node.instruction, False)], rcond), context_map, global_assumptions, equivalence_options)
            # Do NOT set equivalent if this proposition has been shown to be a lemma. Setting 'equivalent' to True again would just recursively re-trigger
            # downstream propositions to be re-proved.
        elif isinstance(left_node, SynchronizationGraph.Phi) and isinstance(right_node, SynchronizationGraph.Phi):
            if proof.is_proven(current_pair): # attempting to complete the inductive step
                l_inductive = left.loop_phi_arguments[(left_node.instruction, False)]
                r_inductive = right.loop_phi_arguments[(right_node.instruction, False)]
                # Do NOT set the 'equivalent' variable here. This lemma has already been assumed equivalent, and if this if-condition is satisfied, then it is provably so.
                # But setting 'equivalent' will trigger a cyclic round of adding the phi's dependencies to the worklist again.
                if equivalent_values(l_inductive, r_inductive, None, context_map, global_assumptions, permissive_typing=True) and equivalent_expressions(
                    loop_continuation_condition(left_node.instruction, left, lcontrol_deps),
                    loop_continuation_condition(right_node.instruction, right, rcontrol_deps),
                    None, context_map, global_assumptions
                ):
                    induction_complete.add(current_pair)
            else: # Show the base cases equivalent. Then we can assume the inductive assumption.
                l_base_case = left.loop_phi_arguments[(left_node.instruction, True)]
                r_base_case = right.loop_phi_arguments[(right_node.instruction, True)]
                equivalent = equivalent_values(l_base_case, r_base_case, None, context_map, global_assumptions, permissive_typing=True)
        else:
            raise ValueError("Mismatched or unexpected node types in synchronizaton proof.")

        if equivalent:
            # At this point, we can make the assumptions in the global assumptions map; that is, we can assume they are true.
            # For efficiency's sake, we re-use the same map as this node's local variable map.
            equivalence_candidates = proof.mark_as_proven(current_pair, context_map, global_assumptions)
            worklist.extend(equivalence_candidates)
    
    ### Check if inductive assumptions were met.
    # Pre-generate a list becase revoke() modifies the adjacency list in the proof object.
    phi_pairs: list[tuple[SynchronizationGraph.Phi, SynchronizationGraph.Phi]] = [
        (l, r) for l, r in proof.adjacency_list # unpacking the tuple here for the typechecker's sake.
        if proof.is_proven((l, r)) and isinstance(l, SynchronizationGraph.Phi) and isinstance(r, SynchronizationGraph.Phi)
    ]
    for current_pair in phi_pairs:
        # We need to revoke lemmas that meet the following requirements:
        # 1. The inductive assumption was made: this pair has been proven (is a lemma, not just a conjecture). 
        #    If the pair is just a conjecture, then the inductive assumption was never made, and thus there is thus nothing to revoke.
        # 2. This pair's induction was NOT completed. We manually track all pairs for which induction was completed.
        # This code also implicitly handles the situation where this pair was removed as part of an earlier incutive-assumption-correction iteration.
        if proof.is_proven(current_pair) and current_pair not in induction_complete:
            proof.revoke(current_pair)

    # We also need to revoke bad global assumptions. If we assume that two global variables are equivalent but these globals end up storing different 
    # values at the end of the function, then the assumption was incorrect and we should revoke anything proven based on those faulty assumptions.
    #
    # We do this after proving as much as we can. This is because we'll need a variable mapping in order to prove the contents of some global variables 
    # equivalent, and the proof process generates the variable mapping.
    proof.revoke_incorrect_global_assumptions(
        left.return_stack, right.return_stack,
        {s.name: var for s, var in left.global_progenitors.items()},
        {s.name: var for s, var in right.global_progenitors.items()},
        left.global_derivation, right.global_derivation
    )

    return proof

def prove_equivalence(left: Execution, right: Execution, proof: SynchronizationProof, equivalence_options: EquivalenceOptions) -> str | None:
    """Prove that two executions are equivalent. Returns None if the executions are equivalent and a string describing the reason for nonequivalence if not.
    """

    pair_names = equivalence_options.pair_names

    ### Check that each call has a corresponding equivalent call in the other function.
    # Run this before call consistency checking because it's faster (equivalence for each function has already
    # been computed) and because we can get better error messages than out of consistency checking.
    call_mapping = proof.equivalent_calls()
    for order, fn in ((0, left.fn), (1, right.fn)):
        for bb in fn.basic_blocks:
            for instruction in bb:
                if isinstance(instruction.op, FunctionCall):
                    # This call has no corresponding equivalent call in the other function. This is a 
                    # necessary condition for equivalence, so we short-circuit and return False here.
                    if len(call_mapping.get(instruction, ())) == 0:
                        return f"Function {instruction.op.fname} in the {pair_names[order]} function has no equivalent in the {pair_names[1 - order]} function."

    ### Function name consistency checking
    left_calls = [ins for bb in left.fn for ins in bb if isinstance(ins.op, FunctionCall)]
    right_calls = [ins for bb in right.fn for ins in bb if isinstance(ins.op, FunctionCall)]
    # Assign each call a variable
    callvars: dict[SSAInstruction, z3.ArithRef] = {}
    for fn_id, calls in (("l", left_calls), ("r", right_calls)):
        callid = 0
        for call in calls:
            callvars[call] = z3.Int(f"{fn_id}_{callid}")
            callid += 1

    constraints: list[z3.BoolRef | bool] = []
    # Within a given function, determine which calls have the same name and which do not.
    for calls in (left_calls, right_calls):
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                if calls[i].op.fname == calls[j].op.fname: # type: ignore -- typechecker does not recognize that calls[i].op is a FunctionCall
                    constraints.append(callvars[calls[i]] == callvars[calls[j]])
                else:
                    constraints.append(callvars[calls[i]] != callvars[calls[j]])
    
    # Equivalent instructions should have consistent function names. Note that if a given function aligns with multiple other functions, then its name
    # must be consistent with at least one other function. But it is not necessary that it is consistent with all of them; and in general that will not
    # be true. Consider the following two programs:
    # void left() {    void right()
    #     foo(2);          fizz(2);   
    #     bar(2);          buzz(2);
    # }                }
    # Because all four functions here have equivalent arguments, they will all be shown equivalent when
    # building the product program. These are consistent: in fact, there are multiple possible consistent
    # assignments: (foo == fizz and bar == buzz) or (foo == buzz and bar == fizz). This is the reason for 
    # the z3.Or constraint below.
    for call_instruction, alternatives in call_mapping.items():
        focus_var = callvars[call_instruction] # type: ignore -- because the type checker doesn't know that call_instruction.op is a FunctionCall
        if len(alternatives) == 1:
            constraints.append(focus_var == callvars[alternatives[0]]) # type: ignore -- because the type checker doesn't know that alternatives[i].op is a FunctionCall
        else:
            constraints.append(z3.Or(*(focus_var == callvars[alt] for alt in alternatives))) # type: ignore

    if unsatisfiable(z3.And(*constraints)): # type: ignore -- Imprecise z3 typing
        return "Function names are inconsistent."
    
    ### Function argument heapspace checking
    lheap = left.return_heap
    rheap = right.return_heap
    def flatten_arglist(execution: Execution) -> list[AddressableValue[Symbol]]:
        """Build an argument list by flattening each struct argument into a sequence of its components."""
        arguments: list[AddressableValue[Symbol]] = []
        for argument in execution.arguments:
            if isinstance(argument, CompoundValue):
                last_offset = -1
                for offset, field in argument:
                    assert offset > last_offset # the struct insertion-ordering-preserving behavior of dictionaries should guarantee this and CompoundValues are usually built in order---but it's good to check.
                    assert isinstance(field, AddressableValue) and isinstance(field.base_address, Symbol) # TODO: support nested struct arguments.
                    arguments.append(field)
                    last_offset = offset
            else:
                arguments.append(argument)
        return arguments
    
    var_map = proof.build_full_var_map()

    larguments = flatten_arglist(left)
    rarguments = flatten_arglist(right)
    global_assumptions = GlobalAssumptions(left.globals, right.globals, proof.symbol_factory)
    for i, larg, rarg in zip(range(len(larguments)), larguments, rarguments):
        if not equivalent_heaplets(lheap, rheap, larg, rarg, var_map, global_assumptions, equivalence_options):
            return f"Heapspace memory accessible from argument {i + 1} is not equivalent."
    
    # Handle extra arguments. We allow extra arguments if they don't have any impact on the function's observable behavior.
    extra_arg_message = "Heapspace memory accessible from extra argument in {} has observable modifications."
    if len(larguments) > len(rarguments):
        nright = len(rarguments)
        for larg in larguments[nright:]:
            if larg.base_address in lheap.mapping:
                return extra_arg_message.format(pair_names[0])
    elif len(rarguments) > len(larguments):
        nleft = len(larguments)
        for rarg in rarguments[nleft:]:
            if rarg.base_address in rheap.mapping:
                return extra_arg_message.format(pair_names[1])

    ### Return value equivalence checking, include reachable heapspace.
    lrv = left.return_value
    rrv = right.return_value

    # Heapspace reachability
    if isinstance(lrv, AddressableValue) and isinstance(rrv, AddressableValue):
        if isinstance(lrv.base_address, Symbol) and isinstance(rrv.base_address, Symbol):
            if not equivalent_heaplets(lheap, rheap, lrv, rrv, var_map, global_assumptions, equivalence_options):
                return "Heapspace memory reachable from the return values is not equivalent."
    elif lrv is not None and rrv is not None and (isinstance(lrv, AddressableValue) or isinstance(rrv, AddressableValue)):
        return "Return values have mixed addressability (ability to represent a valid memory address)."

    # Contents of the heap value.
    if lrv is not None and rrv is not None:
        equivalent = equivalent_values(lrv, rrv, None, var_map, global_assumptions, permissive_typing=True)
    elif equivalence_options.ignore_mixed_return_behavior:
        equivalent = True
    else:
        equivalent = lrv == rrv
    if not equivalent:
        return f"Return values are nonequivalent."
    
    ### Global variable consistency checking. We must do this at the end, after all assumptions have been made.
    # Process generally mirrors the function name consistency checking above.
    constraints: list[z3.BoolRef | bool] = []
    lglobals: list[str] = [g.name for g in left.global_progenitors]
    rglobals: list[str] = [g.name for g in right.global_progenitors]
    # Assign each global variable a variable
    gvars: dict[str, z3.ArithRef] = {}
    for fn_id, gs in (("l", lglobals), ("r", rglobals)):
        for i, varname in enumerate(gs):
            gvars[varname] = z3.Int(f"{fn_id}_{i}")

    # Different global variables are different.
    for gs in (lglobals, rglobals):
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                constraints.append(gvars[gs[i]] != gvars[gs[j]])

    # Determine which globals map to which other globals
    lequivalents: dict[str, list[str]] = {}
    requivalents: dict[str, list[str]] = {}
    for global_pair in itertools.product(lglobals, rglobals):
        if global_pair in var_map or global_pair in global_assumptions:
            lg, rg = global_pair
            # Map each derived global symbol to the root progenitor symbol it was derived from.
            # If G aligns with H, then G[8] must align with H[8], and not some other derived symbol like J[8].
            lg = left.global_derivation[lg]
            rg = right.global_derivation[rg]
            if lg in lequivalents:
                lequivalents[lg].append(rg)
            else:
                lequivalents[lg] = [rg]
            if rg in requivalents:
                requivalents[rg].append(lg)
            else:
                requivalents[rg] = [lg]
    
    # Each global must map to at least one other global in its equivalence class.
    for var, equivalents in itertools.chain(lequivalents.items(), requivalents.items()):
        if len(equivalents) == 1:
            constraints.append(gvars[var] == gvars[equivalents[0]])
        else:
            constraints.append(z3.Or(*(gvars[var] == gvars[eq] for eq in equivalents))) # type: ignore

    # TODO: check to see that equivalent globals have equivalent values.

    if unsatisfiable(z3.And(*constraints)): # type: ignore -- Imprecise z3 typing
        return "Global variables are inconsistent"
    return None
