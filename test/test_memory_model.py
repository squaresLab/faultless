"""Test the memory model operations individually
"""

import unittest

import z3

from .utils import TestValues
from faultless.ir import *
from faultless.c import PRIMITIVE_TYPES

_float = PRIMITIVE_TYPES["float"]
_double = PRIMITIVE_TYPES["double"]
_char = PRIMITIVE_TYPES["char"]
_int = PRIMITIVE_TYPES["int"]
_long = PRIMITIVE_TYPES["long"]
_unk = UnknownType()

z3solver = z3.Solver()
class TestMemoryModelOperations(TestValues):
    def inductive_offset(self, index: SymbolicExpression, induction_var: SymbolicExpression, base_case: SymbolicExpression | int, update: SymbolicExpression, condition: z3.BoolRef, read_size: int, solver: z3.Solver = z3.Solver()) -> Offset:
        """Build an inductive offset from the information provided. If possible, this will be represented as an instance of Offset
        rather than InductiveOffset, which is desirable because Offset does not require defining any potentially slow recursive functions.
        """
        # The update is the coefficient because it is repeatedly added to the looping variable. Loops of the form
        #   for (int i = b; i < n; i += u)
        # can be represented by the equation i = u*phi + b, where phi represents the loop iteration. 
        # Loops that are of this form pass the if-check below.
        coefficient: SymbolicExpression = z3.simplify(update - induction_var) # type: ignore
        if induction_var in z3.z3util.get_vars(coefficient):
            # This means update is NOT of the form phi + u where phi is the induction variable.
            # The usually more efficient affine form computed below therefore cannot be used, and we
            # fall back on the more general InductiveOffset.
            return InductiveOffset(index, induction_var, base_case, update, condition, read_size)
        
        # Because the coefficient can be any symbolic expression, it may not always be the case that it
        # is always increasing or decreasing (e.g. if it is an unconstrained symbolic variable). Therefore,
        # we separately check both cases.

        # z3.Not for validity query
        increasing = unsatisfiable(z3.Not(coefficient > 0), solver) # type: ignore
        decreasing = unsatisfiable(z3.Not(coefficient < 0), solver) # type: ignore

        assert not (increasing and decreasing), f"Invariant issue: {update} is both increasing and decreasing."

        valid_index = induction_var % coefficient == base_case
        if increasing:
            full_condition = z3.And(induction_var >= base_case, valid_index, condition)
        elif decreasing:
            full_condition = z3.And(induction_var <= base_case, valid_index, condition)
        else: # We can make no claims about the induction var's relationship to the base case.
            full_condition = z3.And(valid_index, condition)
        return Offset(index, full_condition, read_size) # type: ignore


    def test_single_cell_read(self):
        """Tests reading from an empty heap.
        """
        space = Heap()
        fresh_val = z3.BitVec('x[0]', 32)
        self.assertValuesEqual(
            space.read(Symbol(Pointer(_int), 'x', z3.Int('x'), False), Offset(0, True, 4), _int),
            AddressableValue(_int, fresh_val, Symbol(_int, 'x[0]', fresh_val, False), ())
        )

    def test_single_cell_write_reads(self):
        """Tests writing a single value to the heap, then reading from it and elsewhere in the heap.

        *ptr = 3;
        ptr[1];
        *ptr;
        """
        space = Heap()
        base_addr = Symbol(Pointer(_int), 'ptr', z3.Int('ptr'), False)
        space.write(base_addr, Offset(0, True, 4), Value(_int, z3.BitVecVal(3, 32)))

        # Read a different value from the one we wrote.
        fresh_val = z3.BitVec('ptr[4]', 32)
        self.assertValuesEqual(
            space.read(base_addr, Offset(4, True, 4), _int),
            AddressableValue(_int, fresh_val, Symbol(_int, 'ptr[4]', fresh_val, False), ())
        )

        # Read the value we wrote earlier
        self.assertValuesEqual(
            space.read(base_addr, Offset(0, True, 4), _int),
            Value(_int, z3.BitVecVal(3, 32))
        )

    def test_read_write_same_condition(self):
        """Tests writing then reading to possibly different elements under the same path conditions
        
        if (x < 4) {
            y[0] = 7;
            print(y[x]);
        }
        """
        space = Heap()
        y = Symbol(Pointer(_char), 'y', z3.Int('y'), False)
        x = z3.Int('x')
        space.write(y, Offset(0, x < 4, 1), Value(_char, z3.BitVecVal(7, 8)))

        self.assertValuesEqual(
            space.read(y, Offset(x, x < 4, 1), _char),
            Value(_char, z3.If(x == 0, 7, z3.BitVec("y[x]", 8))) # type: ignore
        )

    def test_refined_read_condition(self):
        """Tests reading from the heap after writing in a less constrained environment.

        a[0] = 7;
        if (x == 0) print(a[x])
        """
        space = Heap()
        a = Symbol(Pointer(_char), 'a', z3.Int('a'), False)
        x = z3.Int('x')
        space.write(a, Offset(0, True, 1), Value(_char, z3.BitVecVal(7, 8)))

        self.assertValuesEqual(
            space.read(a, Offset(x, x == 0, 1), _char),
            Value(_char, z3.BitVecVal(7, 8))
        )

    
    def test_fixed_array_write_single_read(self):
        """Tests writing to the heap in a loop with a constant bound, and then reading from it and elsewhere in the heap.

        for (int i = 0; i < 26; i++) str[i] = i + 65
        str[2];
        str[27];
        """
        space = Heap()
        base_addr = Symbol(Pointer(_char), 'str', z3.Int('str'), False)
        loop_induction_var = z3.BitVec('\\phi_i', _char.size * 8) # Technically there phi_i should be an int and there should be an integer promotion/implicit cast here, but it doesn't matter for the heap.
        # Note: in practice, loop_induction_var would actually be addressable. That doesn't matter for testing the heap searching logic though.
        space.write(base_addr, 
            # Offset(loop_induction_var, loop_induction_var < 26, 1), 
            InductiveOffset(loop_induction_var, loop_induction_var, z3.BitVecVal(0, _char.size * 8), loop_induction_var + 1, loop_induction_var < 26, 1),
            Value(_int, loop_induction_var + 65)
        )

        # Within the written segment
        self.assertValuesEqual(
            space.read(base_addr, Offset(z3.BitVecVal(2, _char.size * 8), True, 1), _char),
            Value(_char, loop_induction_var + 65)
        )

        # Outside of the written segment
        fresh_var = z3.BitVec('str[27]', _char.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, Offset(z3.BitVecVal(27, _char.size * 8), True, 1), _char),
            AddressableValue(_char, fresh_var, Symbol(_char, 'str[27]', fresh_var, False), ())
        )
    
    def test_bounded_array_write_group_read(self):
        """Tests writing to the heap in a loop with a constant bound, and then reading from it in a downstream loop.

        for (int i = 0; i < 26; i++) str[i] = i + 65
        for (int j = 0; j < 26; j++) str[j]
        for (int j = 0; j < 27; j++) str[j]
        """
        space = Heap()
        base_addr = Symbol(Pointer(_char), 'str', z3.Int('str'), False)
        loop_induction_var = z3.BitVec('\\phi_i', _char.size * 8) # Technically there \\phi_i should be an int and there should be an integer promotion/implicit cast here, but it doesn't matter for the heap.
        # Note: in practice, loop_induction_var would actually be addressable. That doesn't matter for testing the heap searching logic though.
        space.write(base_addr,
            self.inductive_offset(loop_induction_var, loop_induction_var, z3.BitVecVal(0, _char.size * 8), loop_induction_var + 1, loop_induction_var < 26, 1),
            Value(_char, loop_induction_var + 65)
        )

        # Read only values that were previously written
        query_loop_var = z3.BitVec('\\phi_j',_char.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, self.inductive_offset(query_loop_var, query_loop_var, z3.BitVecVal(0, _char.size * 8), query_loop_var + 1, query_loop_var < 26, _char.size), _char),
            Value(_char, loop_induction_var + 65)
        )

        # Read past the end of the values that were previously written
        fresh_var = z3.BitVec('str[\\phi_j]', _char.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, self.inductive_offset(query_loop_var, query_loop_var, 0, query_loop_var + 1, query_loop_var < 27, _char.size), _char),
            Value(_char, z3.If(z3.Exists([loop_induction_var], z3.And(0 <= loop_induction_var, loop_induction_var < 26, loop_induction_var == query_loop_var)), loop_induction_var + 65, fresh_var)), # type: ignore
        )

    def test_unbounded_array_write_single_read(self):
        """Tests writing to the heap in a loop with a variable bound, and then reading from it.

        for (int i = 0; i < n; i++) vals[i] = i;
        vals[2];
        """
        space = Heap()
        base_addr = Symbol(Pointer(_int), 'vals', z3.Int('vals'), False)
        loop_induction_var = z3.BitVec('\\phi_i', _int.size * 8)
        loop_max_iters = z3.BitVec('n', _int.size * 8)
        space.write(base_addr, self.inductive_offset(4 * loop_induction_var, loop_induction_var, z3.BitVecVal(0, _int.size * 8), loop_induction_var + 1, loop_induction_var < loop_max_iters, 4), Value(_int, loop_induction_var))

        self.assertValuesEqual(
            space.read(base_addr, Offset(z3.BitVecVal(8, _int.size * 8), True, 4), _int),
            Value(_int, z3.If(z3.Exists([loop_induction_var], z3.And(0 <= loop_induction_var, loop_induction_var < loop_max_iters, 4 * loop_induction_var == 8)), loop_induction_var, z3.BitVec(f"vals[8]", _int.size * 8))), # type: ignore
        )

    def test_unbounded_array_write_unbounded_read(self):
        """Tests writing to the heap in a loop with a variable bound, then reading from it with a variable bound.
        
        for (int i = 0; i < n; i++) vals[i] = i;
        for (int j = 0; j < n; i++) vals[i]
        for (int j = 0; j < m; i++) vals[i]
        """
        space = Heap()
        base_addr = Symbol(Pointer(_int), 'vals', z3.Int('vals'), False)
        write_induction_var = z3.BitVec('\\phi_i', _int.size * 8)
        max_elements_n = z3.BitVec('n', _int.size * 8)
        space.write(base_addr,
            self.inductive_offset(write_induction_var, write_induction_var, 0, write_induction_var + 1, write_induction_var < max_elements_n, 4),
            Value(_int, write_induction_var)
        )

        query_loop_var = z3.BitVec('\\phi_j', _int.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, self.inductive_offset(query_loop_var, query_loop_var, z3.BitVecVal(0, query_loop_var.size()), query_loop_var + 1, query_loop_var < max_elements_n, _int.size), _int), 
            Value(_int, write_induction_var)
        )

        max_elements_m = z3.BitVec('m', _int.size * 8)
        fresh_var = z3.BitVec('vals[\\phi_j]', _int.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, self.inductive_offset(query_loop_var, query_loop_var, z3.BitVecVal(0, query_loop_var.size()), query_loop_var + 1, query_loop_var < max_elements_m, _int.size), _int), 
            Value(_int, z3.If(z3.Exists([write_induction_var], z3.And(0 <= write_induction_var, write_induction_var < max_elements_n, write_induction_var == query_loop_var)), write_induction_var, fresh_var)) # type: ignore
        )

    def test_mismatching_strides(self):
        """Tests writing to the heap in a loop with an increment of 2, and then reading from the offsets that weren't written to.

        for (int i = 0; i < n; i += 2) vals[i] = i;
        for (int i = 1; i < n; i += 2) vals[i];
        """
        space = Heap()
        base_addr = Symbol(Pointer(_int), 'vals', z3.Int('vals'), False)
        write_induction_var = z3.BitVec('\\phi_i', _int.size * 8)
        loop_max_iters = z3.BitVec('n', _int.size * 8)
        space.write(base_addr,
            self.inductive_offset(write_induction_var, write_induction_var, z3.BitVecVal(0, write_induction_var.size()), write_induction_var + 2, write_induction_var < loop_max_iters, _int.size * 8),
            Value(_int, write_induction_var)
        )

        read_induction_var = z3.BitVec('\\phi_j', _int.size * 8)
        fresh_var = z3.BitVec('vals[\\phi_j]', _int.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, self.inductive_offset(read_induction_var, read_induction_var, z3.BitVecVal(1, read_induction_var.size()), read_induction_var + 2, read_induction_var < loop_max_iters, _int.size * 8), _int),
            AddressableValue(_int, fresh_var, Symbol(_int, 'vals[\\phi_j]', fresh_var, False), ())
        )

    def test_pre_loop_write(self):
        """Tests reading from the heap in a loop where a write was performed before the loop

        a[3] = 7
        for (int i = 0; i < n; ++i) a[i];
        """
        space = Heap()
        a = Value.make((Pointer(_char), "a"))
        space.write(a.base_address, Offset(z3repr(IntegerConstant(3, _int)), True, 1), Value.make(IntegerConstant(7, _char)))
        
        phi_i = Value.make((_int, "\\phi_i"))
        fresh_var = Value.make((_char, "a[\\phi_i]"))
        loop_max_iters = Value.make((_int, "n"))
        self.assertValuesEqual(
            space.read(a.base_address, self.inductive_offset(phi_i.expr, phi_i.expr, z3repr(IntegerConstant(0, _int)), phi_i.expr + 1, phi_i.expr < loop_max_iters.expr, 1), _char),
            Value(_char, z3.If(phi_i.expr == 3, z3repr(IntegerConstant(7, _char)), fresh_var.expr)) # type: ignore
        )

    def test_sequence_of_constant_writes(self):
        """Tests reading from a heap after a sequence of constant writes.

        a[0] = 5;
        a[1] = 10;
        a[2] = 15;
        a[x];
        """
        space = Heap()
        base_addr = Symbol(Pointer(_char), 'a', z3.Int('a'), False)
        space.write(base_addr, Offset(0, True, 1), Value(_char, z3.BitVecVal(5, _char.size * 8)))
        space.write(base_addr, Offset(1, True, 1), Value(_char, z3.BitVecVal(10, _char.size * 8)))
        space.write(base_addr, Offset(2, True, 1), Value(_char, z3.BitVecVal(15, _char.size * 8)))

        x = z3.Int('x')
        fresh_var = z3.BitVec('a[x]', _char.size * 8)
        self.assertValuesEqual(
            space.read(base_addr, Offset(x, True, 1), _char),
            Value(_char, z3.If(x == 0, z3.BitVecVal(5, _char.size * 8), z3.If(x == 1, z3.BitVecVal(10, _char.size * 8), z3.If(x == 2, z3.BitVecVal(15, _char.size * 8), fresh_var)))) # type: ignore
        )

    def test_struct_write_and_read(self):
        """Tests writing a struct to memory, then reading it back.
        
        struct point pt = { .x=3, .y=5 };
        pt
        """
        space = Stack()

        point_t = Struct("point", [UDT.Field(_int, "x"), UDT.Field(_int, "y")])
        base_addr = Variable(point_t, "pt")
        value = CompoundValue(point_t, {0: Value.make(IntegerConstant(3, _int)), 4: Value.make(IntegerConstant(5, _int))})

        space.write(base_addr, Offset(0, True, 8), value)
        self.assertValuesEqual(
            space.read(base_addr, Offset(0, True, 8), point_t),
            value
        )
        

class TestHeapState(TestValues):
    def test_single_write(self):
        """Tests writing a single value to the heap.
        """
        candidate = Heap()

        x_var = z3.BitVec('x', _int.size * 8)
        base_addr = Symbol(_int, "x", x_var, False)
        candidate.write(base_addr, Offset(0, True, 4), Value(_int, z3.BitVecVal(5, _int.size * 8)))

        mapping: dict[Symbol, AddressSet] = {
            base_addr: Write(Offset(0, True, 4), Value(_int, z3.BitVecVal(5, _int.size * 8)), None)
        }
        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)

    def test_consecutive_writes(self):
        """Tests writing two values in sequence.
        """
        candidate = Heap()

        x_var = z3.BitVec('x', _int.size * 8)
        base_addr = Symbol(_int, "x", x_var, False)
        candidate.write(base_addr, Offset(0, True, 4), Value(_int, z3.BitVecVal(5, _int.size * 8)))
        candidate.write(base_addr, Offset(0, True, 4), Value(_int, z3.BitVecVal(9, _int.size * 8)))

        mapping: dict[Symbol, AddressSet] = {
            base_addr: Write(Offset(0, True, 4), Value(_int, z3.BitVecVal(9, _int.size * 8)), Write(Offset(0, True, 4), Value(_int, z3.BitVecVal(5, _int.size * 8)), None))
        }
        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)

    def test_independent_writes(self):
        """Tests writing two values to different base addresses.
        """
        candidate = Heap()

        x_var = z3.BitVec('x', _int.size * 8)
        x = Symbol(_int, "x", x_var, False)
        y_var = z3.BitVec('y', _int.size * 8)
        y = Symbol(_int, "y", y_var, False)
        candidate.write(x, Offset(0, True, 4), Value(_int, z3.BitVecVal(3, _int.size * 8)))
        candidate.write(y, Offset(0, True, 4), Value(_int, z3.BitVecVal(7, _int.size * 8)))

        mapping: dict[Symbol, AddressSet] = {
            x: Write(Offset(0, True, 4), Value(_int, z3.BitVecVal(3, _int.size * 8)), None),
            y: Write(Offset(0, True, 4), Value(_int, z3.BitVecVal(7, _int.size * 8)), None)
        }
        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)
    
    def test_struct_write(self):
        candidate = Heap()

        struct_t = Struct("point", [UDT.Field(INTEGER, "x"), UDT.Field(INTEGER, "y")])
        value = CompoundValue.make((struct_t, "addr"))
        addr = Value.make((Pointer(struct_t), "addr")).base_address
        candidate.write(addr, Offset(0, True, 4), value)

        mapping: dict[Symbol, AddressSet] = {
            addr: Write(Offset(4, True, 4), Value.make((_int, 'addr[4]')), Write(Offset(0, True, 4), Value.make((_int, 'addr[0]')), None))
        }

        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)
    
    def test_simple_linked_list(self):
        """Tests induction for a simple while loop that initializes a linked list.

        struct node { int val; struct node * next;} * n;
        head = n;
        while (n) {
            n->val = -1;
            n = n->next;
        }
        // We also check these two values.
        head->val;
        head->next;
        """
        node_star = Pointer(IncompleteStruct("node"))
        node_t = Struct("node", [UDT.Field(_int, "val"), UDT.Field(node_star, "next")])
        candidate = Heap()

        n_var = z3.Int("n")
        n = Symbol(Pointer(node_t), "n", n_var, False)
        phi_i_var = z3.Int("\\phi_i")
        phi_i = Symbol(Pointer(node_t), "\\phi_i", phi_i_var, True)
        end_var = z3.Int("\\$phi_i")

        # Loop body
        candidate.init_var_induction(n, phi_i, z3.IntVal(0))
        int_neg_one = Value(_int, z3.BitVecVal(-1, _int.size * 8))
        candidate.write(phi_i, Offset(0, phi_i_var != 0, 4), int_neg_one)
        candidate.read(phi_i, Offset(8, phi_i_var != 0, 8), node_star)

        # Loop body oracle
        mapping: dict[Symbol, AddressSet] = {
            phi_i: Write(Offset(0, phi_i_var != 0, 4), int_neg_one, None)
        }
        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)

        # Post loop
        head_val = candidate.read(n, Offset(0, end_var == 0, 4), _int)
        head_next = candidate.read(n, Offset(8, end_var == 0, 8), node_star)

        # Post-loop oracle
        n_8_var = z3.Int("n[8]")
        n_8 = Symbol(node_star, "n[8]", n_8_var, False)
        self.assertValuesEqual(head_val, int_neg_one)
        self.assertValuesEqual(head_next, AddressableValue(node_star, n_8_var, Symbol(node_star, "n[8]", n_8_var, False), ()))
        mapping[n] = Write(Offset(0, end_var == 0, 4), int_neg_one, None)
        self.assertHeapStorageEqual(candidate, reference)

    def test_simple_linked_list_with_base_offset(self):
        """Tests induction for a simple while loop that initializes a linked list.

        struct node { int val; struct node * next;} * input;
        head = n; //perhaps this is an array of linked lists.
        n++; // Now n points to the second linked list in the array.
        while (n) {
            n->val = 7;
            n = n->next;
        }

        head->val; // should be a fresh val
        head->next;
        head++;
        head->val; // should be 7
        head->next;
        """
        inc_node_star = Pointer(IncompleteStruct("node"))
        node_t = Struct("node", [UDT.Field(_int, "val"), UDT.Field(inc_node_star, "next")])
        node_ptr = Pointer(node_t)
        candidate = Heap()

        n = Value.make((node_ptr, "n"))
        phi = Value.make((node_ptr, "\\phi_n"))
        phi.base_address.is_induction_var=True
        post_phi = Value.make((node_ptr, "\\post_phi"))
        seven = Value.make(IntegerConstant(7, _int))

        # Execute the loop
        candidate.init_var_induction(n.base_address, phi.base_address, z3.IntVal(16))
        candidate.write(phi.base_address, Offset(0, phi.expr != 0, 4), seven)
        candidate.read(phi.base_address, Offset(8, phi.expr != 0, 8), node_ptr)

        # Check that the loop wrote the correct things.
        mapping: dict[Symbol, AddressSet] = {
            phi.base_address: Write(Offset(0, phi.expr != 0, 4), seven, None)
        }
        reference = Heap(mapping)
        self.assertHeapStorageEqual(candidate, reference)

        ### Now invoke inductive modifications post-loop
        # The first read hits fresh memory because that part of memory was not touched in the loop.
        head_val = candidate.read(n.base_address, Offset(z3.IntVal(0), post_phi.expr == 0, 4), _int)
        head_next = candidate.read(n.base_address, Offset(z3.IntVal(8), post_phi.expr == 0, 8), node_ptr)
        self.assertValuesEqual(head_val, Value.make((_int, "n[0]")))
        self.assertValuesEqual(head_next, Value.make((node_ptr, "n[8]")))

        # now increment head and try again
        head_val = candidate.read(n.base_address, Offset(z3.IntVal(16), post_phi.expr == 0, 4), _int)
        head_next = candidate.read(n.base_address, Offset(z3.IntVal(24), post_phi.expr == 0, 8), node_ptr)
        self.assertValuesEqual(head_val, seven)
        self.assertValuesEqual(head_next, Value.make((node_ptr, "n[24]")))

        mapping[n.base_address] = Write(Offset(z3.IntVal(16), post_phi.expr == 0, 4), seven, None)
        self.assertHeapStorageEqual(candidate, reference)