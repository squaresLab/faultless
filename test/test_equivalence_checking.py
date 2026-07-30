"""Checks the top-level APIs for determining if two functions are equivalent or not.

This test suite implicitly tests the entire pipeline.
"""

import unittest

from faultless import are_equivalent, EquivalenceOptions, z3repr_options


class TestEquivalenceChecking(unittest.TestCase):
    def test_simple_returns(self):
        ident = "int ident(int x) { return x; }"
        increment = "int increment(int y) { return y + 1; }"
        plusplus = "int plusplus(int z) { return ++z; }"

        self.assertTrue(are_equivalent(increment, plusplus))
        self.assertFalse(are_equivalent(ident, increment))

    def test_for_loops(self):
        plusplus = """void printn(int n) { for (int i = 0; i < n; ++i) { print(i); }}"""
        plus_one = """void printn(int n) { for (int i = 0; i < n; i += 1) { print(i); }}"""
        plus_two = """void printn(int n) { for (int i = 0; i < n; i += 2) { print(i); }}"""

        self.assertTrue(are_equivalent(plusplus, plus_one))
        self.assertFalse(are_equivalent(plusplus, plus_two))

    def test_return_behavior(self):
        retone = """int foo() { return 1; }"""
        noop = """void noop(int x) { }"""
        extraneous = """void extraneous(int x) { x = x + 1; }"""

        self.assertTrue(are_equivalent(noop, extraneous))
        self.assertFalse(are_equivalent(retone, noop))

    def test_loop_with_function_and_multi_function_prefix(self):
        candidate = """
        void foo(int x, int y, int n) {
            int b = init(x, 0);
            int a = doit(x, y);
            bar(a, b + 1);
            for (int i = 0; i < n; ++i) {
                i = baz(i);
            }
        }
        """

        reference = """
        void foo(int x, int y, int n) {
            int a = doit(x, y);
            int b = init(x, 0) + 1;
            bar(a, b);
            for (int i = 0; i < n; i += 1) {
                i = baz(i);
            }
        }
        """
        self.assertTrue(are_equivalent(candidate, reference))
        self.assertFalse(are_equivalent(candidate, reference.replace("+= 1", "+= 2")))

    def test_cross_equivalence_cluster_function_name_consistency(self):
        foobar = """void foobar(int x) { foo(x); bar(x); }"""
        fizzbuzz = """void fizzbuzz(int x) { fizz(x); bar(x); }"""
        baz2 = """void baz2(int x) { baz(x); baz(x); }"""

        self.assertTrue(are_equivalent(foobar, fizzbuzz))
        self.assertFalse(are_equivalent(foobar, baz2))

    def test_dependency_separated_function_name_consistency(self):
        func1 = """
        void thing(int n) {
            foo(n);
            for (int i = 0; i < n; ++i) {
                bar(i);
            }
        }
        """

        func2 = """
        void other(int n) {
            baz(n);
            for (int i = 0; i < n; ++i) {
                biz(i);
            }
        }
        """

        self.assertTrue(are_equivalent(func1, func2))
        self.assertFalse(are_equivalent(func1, func2.replace("biz", "baz")))

    def test_var_mapping_function_pairs(self):
        foobar = """void foobar(int x) { bar(foo(x)); }"""
        fizzbuzz = """void fizzbuzz(int y) { buzz(fizz(y)); }"""
        barbar = """void barbar(int x) { bar(bar(x)); }"""

        self.assertTrue(are_equivalent(foobar, fizzbuzz))
        self.assertFalse(are_equivalent(foobar, fizzbuzz.replace("(y)", "(y + 1)")))
        self.assertFalse(are_equivalent(foobar, barbar))
    
    def test_var_mapping_sequential_loops(self):
        ij = """
        int ij(int n) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += i;
            }
            sum = sum * 2;
            for (int j = 1; j < n; j = j + 2) {
                sum -= j;
            }
            return sum;
        }
        """

        ab = """
        int ab(int n) {
            int total = 0;
            for (int a = 0; a < n; ++a) {
                total += a;
            }
            total = total * 2;
            for (int b = 1; b < n; b = b + 2) {
                total -= b;
            }
            return total;
        }
        """

        self.assertTrue(are_equivalent(ij, ab))

    def test_function_call_in_loop_head(self):
        clearbuffer = """void clear_buffer() { while (getchar() != -1); }"""
        self.assertTrue(are_equivalent(clearbuffer, clearbuffer))

    def test_differentiating_control_flow(self):
        withif = """void withif(int x, int y) { if (x > 0) foo(x, y); }"""
        noif = """void noif(int x, int y) { foo(x, y); }"""
        elsecall = """void elsecall(int x, int y) { if (x <= 0) {} else { foo(x, y); } } """

        self.assertTrue(are_equivalent(withif, elsecall))
        self.assertFalse(are_equivalent(withif, noif))

    def test_constant_address_heap_equivalence(self):
        twowrites = """int * twowrites(int *a) { a[0] = 9; a[2] = 11; foo(a); return a; }"""
        plustwo = """int * plustwo(int *a) { a[0] = 9; a[2] = a[0] + 2; foo(a); return a; }"""
        diffwrites = """int * diffwrites(int *a) { a[0] = 7; a[2] = 11; foo(a); return a; }"""
        overwrite = "int * overwrite(int *a) { a[0] = 9; a[0] = 11; foo(a); return a; }"
        threewrites = """int * twowrites(int *a) { a[0] = 9; a[1] = 10; a[2] = 11; foo(a); return a; }"""

        self.assertTrue(are_equivalent(twowrites, plustwo))
        self.assertFalse(are_equivalent(twowrites, diffwrites))
        self.assertFalse(are_equivalent(twowrites, overwrite))
        self.assertFalse(are_equivalent(twowrites, threewrites))

    def test_extra_arguments(self):
        twowrites = """void twowrites(int *a) { a[0] = 9; a[2] = 11; }"""
        benignextra = """void benignextra(int *a, int b) {a[0] = 9; b++; a[2] = 11;}"""
        writtenextra = """void writtenextra(int *a, int *b) { a[0] = 9; b[1] = 10; a[2] = 11; }"""

        self.assertTrue(are_equivalent(twowrites, benignextra))
        self.assertFalse(are_equivalent(twowrites, writtenextra))

    def test_returned_struct(self):
        decl = "struct point { int x; int y; };"
        xy = decl + """struct point xy(int a) { struct point pt; pt.x = 3; pt.y = 7; return pt; }"""
        yx = decl + """struct point yx(int a) { struct point pt; pt.y = 7; pt.x = 3; return pt; }"""
        y_only = decl + """struct point y_only(int a) { struct point pt; pt.y = 7; return pt; }"""

        self.assertTrue(are_equivalent(xy, yx))
        self.assertFalse(are_equivalent(xy, y_only))

    def test_struct_value_returned_from_callee(self):
        decls = """
        struct point { int x; int y; };
        struct point getit(int loc);
        """
        one_expression = decls + """int one(long arg) { return getit(arg).y; }"""
        two_expressions = decls + """int two(long arg) { struct point pt = getit(arg); return pt.y; }"""
        diff_field = decls + """int diff(long arg) { return getit(arg).x; }"""

        self.assertTrue(are_equivalent(one_expression, two_expressions))
        self.assertFalse(are_equivalent(one_expression, diff_field))

    def test_string_as_function_argument(self):
        printx = """void printx(int x) { printf("x=%d\\n", x); }"""
        printy = """void printy(int y) { printf("x=%d\\n", y); }"""
        print_long = """void print_long(int z) { printf("x=%ld\\n", (long)z); }"""

        self.assertTrue(are_equivalent(printx, printy))
        self.assertFalse(are_equivalent(printx, print_long))

    def test_element_equivalence(self):
        string = """void foo() { bar("abc"); }"""
        array = """void foo() { char arr[4] = {'a', 'b', 'c', '\\0'}; bar(arr); }"""
        struct = "struct vec { char x; char y; char z; }; void foo() { bar((struct vec){.x='a', .y='b', .z='c'}); };"

        self.assertTrue(are_equivalent(string, array))
        self.assertFalse(are_equivalent(string, struct))

    def test_stack_equivalence(self):
        plus_self = """void plus_self(int x) { int z = x + x; bar(&z); }"""
        times_two = """void times_two(int x) { int y = x * 2; bar(&y); }"""
        plus_one = """void plus_one(int x) { int g = x + 1; bar(&g); }"""

        self.assertTrue(are_equivalent(plus_self, times_two))
        self.assertFalse(are_equivalent(plus_self, plus_one))

    def test_equivalent_derived_values(self):
        malloc2x1 = """int malloc2x1(int x) { int * ptr = malloc(2 * x); return ptr[x + 1]; }"""
        mallocxx1 = """int mallocxx1(int x) { int * ptr = malloc(x + x); return ptr[x + 1]; }"""
        malloc2x2 = """int malloc2x2(int x) { int * ptr = malloc(2 * x); return ptr[x + 2]; }"""

        self.assertTrue(are_equivalent(malloc2x1, mallocxx1))
        self.assertFalse(are_equivalent(malloc2x1, malloc2x2))

    def test_mixed_return_behavior(self):
        voidret = """int foo(int x) { bar(x + 1); return 0; }"""
        valueret = """void foo(int x) { bar(1 + x); }"""

        self.assertTrue(are_equivalent(voidret, valueret, equivalence_options=EquivalenceOptions(ignore_mixed_return_behavior=True)))
        self.assertFalse(are_equivalent(voidret, valueret))

    def test_ignore_extra_arguments(self):
        one = """int main() { start(argv[0]); return 0; }"""
        two = """int main() { start(argv[0], 1); return 0; }"""

        self.assertTrue(are_equivalent(one, two, equivalence_options=EquivalenceOptions(ignore_extra_arguments=True)))
        self.assertFalse(are_equivalent(one, two))
    
    def test_memory_formatting_from_index(self):
        original = """struct s { long l; float f; }; void foo(struct s * ptr) { ptr->l = 4; ptr->f = 2.2; }"""
        decompright = """typedef unsigned long _QWORD; void foo(long long a1) { *(_QWORD *)a1 = 4; *(_QWORD *)(a1 + 8) = 2.2; }"""
        decompwrong = """typedef unsigned long _QWORD; void foo(long long a1) { *(_QWORD *)a1 = 4; *(_QWORD *)(a1 + 4) = 2.2; }"""

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(original, decompright, equivalence_options=EquivalenceOptions(memory_formatting_from_index=0)))
            self.assertFalse(are_equivalent(original, decompwrong, equivalence_options=EquivalenceOptions(memory_formatting_from_index=0)))
    
    def test_stack_initialier_function_inference(self):
        myread = """int myread() { int x; scanf("%d", &x); return x; }"""
        readint = """int readint() { int val; scanf("%d", &val); return val; }"""

        self.assertTrue(are_equivalent(myread, readint, equivalence_options=EquivalenceOptions(infer_stack_initializer_functions=True)))
        self.assertFalse(are_equivalent(myread, readint, equivalence_options=EquivalenceOptions(infer_stack_initializer_functions=False)))

    def test_compound_argument_unpacking(self):
        original = """struct s { char c; float f; }; void foo(struct s x) { bar(x); }"""
        decompiled = """void func2(char a1, float a2) { func7(a1, a2); }"""

        self.assertTrue(are_equivalent(original, decompiled, equivalence_options=EquivalenceOptions(decompose_compound_values_in_parameter_lists=True)))
        self.assertFalse(are_equivalent(original, decompiled, equivalence_options=EquivalenceOptions(decompose_compound_values_in_parameter_lists=False)))

    def test_different_size_returned_bitvecs(self):
        char7 = """char char7() { return 7; }"""
        long7 = """long long7() { return 7; }"""

        self.assertTrue(are_equivalent(char7, long7))

    def test_different_size_arguments(self):
        original = """
        struct s { char c; long l; }; 
        long sum(struct s x, int i) {
            bar(i);
            return x.c + x.l + i;
        }"""
        decompiled = """
        long sum(unsigned long a1, long a2, long a3) {
            bar(a3);
            return a1 + a2 + a3;
        }"""
        
        self.assertTrue(are_equivalent(original, decompiled, equivalence_options=EquivalenceOptions(decompose_compound_values_in_parameter_lists=True)))

    def test_incomplete_types_are_resolved_to_full_types(self):
        original = """
        struct node {int val; struct node * next; };
        void printnextnext(struct node *n) {
            n = n->next;
            print_node(n->next);
        }
        """
        decompiled = """
        typedef unsigned long _QWORD;
        void func3(unsigned long a1) {
            print_node(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8));
        }
        """

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(original, decompiled, equivalence_options=EquivalenceOptions(memory_formatting_from_index=0)))

    def test_joint_dataflow_and_control_flow_cycle(self):
        original = """
        int best_location(int *xs, int n) {
            int best = -1, where = -1;
            for (int i = 0; i < n; i++) {
                if (xs[i] > best) {
                    best = xs[i];
                    where = i;
                }
            }
            return where;
        }
        """

        decompiled = """
        typedef unsigned int _DWORD;
        long func35(long a1, int a2) {
            int i;
            unsigned int v4;
            int v5;

            v5 = -1;
            v4 = -1;
            for (i = 0; i < a2; ++i) {
                if (v5 < *(_DWORD *)(a1 + 4LL * i)) {
                    v5 = *(_DWORD *)(a1 + 4LL * i);
                    v4 = i;
                }
            }
            return v4;
        }
        """

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(original, decompiled, equivalence_options=EquivalenceOptions(memory_formatting_from_index=0)))
    
    def test_numeric_relative_offsets(self):
        original = """struct s { long a; int b; }; struct s * bar(); int foo() { return bar()->b; }"""
        decompiled = """typedef unsigned long _QWORD; int func1() { unsigned long v1 = func5(); return *(_QWORD *)(v1 + 8); }"""

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(original, decompiled))

    def test_no_heap_write_equivalence(self):
        myfree_orig = """void myfree(void * p) { free(p); }"""
        myfree_decomp = """void myfree(unsigned long a1) { free(a1); }"""
        zerostr = """void zerostr(void * p) { *(char *)p = 0; }"""

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(myfree_orig, myfree_decomp))
            self.assertFalse(are_equivalent(myfree_orig, zerostr))

    def test_global_assumptions(self):
        foo = """int foo(int x) { int z = func(x + g); return z - G; }"""
        bar = """int bar(int x) { int z = func(x + h); return z - H; }"""
        baz = """int baz(int x) { int z = func(x + i); return I - z; }"""

        self.assertTrue(are_equivalent(foo, bar))
        self.assertFalse(are_equivalent(foo, baz))
    
    def test_global_variable_consistency(self):
        gh = """void foo() { f(g); f(h); }"""
        gg = """void foo() { f(g); f(g); }"""
        ab = """void bar() { f(a); f(b); }"""

        self.assertTrue(are_equivalent(gh, ab))
        self.assertFalse(are_equivalent(gh, gg))

    def test_only_equivalent_globals_are_mapped(self):
        foo = """void foo() { f(a); g(b, 1); b = -1; }"""
        bar = """void bar() { f(c); g(d, 1); d = -1; }"""
        baz = """void baz() { f(x); g(y, 1); x = -1; }"""
        fiz = """void fiz() { f(x); g(y, 1); }"""
        buz = """void buz() { f(z); g(w, 1); w = 92; }"""

        self.assertTrue(are_equivalent(foo, bar))
        self.assertFalse(are_equivalent(foo, baz))
        self.assertFalse(are_equivalent(foo, fiz))
        self.assertFalse(are_equivalent(foo, buz))

    def test_derived_symbols_with_future_dependent_indices(self):
        foo = """void foo(int *x, int n) { for (int i = 0; i < n; ++i) print(x[i]); }"""
        bar = """void bar(int *y, int n) { for (int j = 0; j < n; ++j) print(y[j]); }"""
        baz = """void baz(int *z, int n) { for (int k = 0; k < n; j += 2) print(z[k]); }"""

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(foo, bar))
            self.assertFalse(are_equivalent(foo, baz))

    def test_derived_symbols_with_recursive_dependent_indices(self):
        foo = """
        void foo(int **x, int n) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j)
                    print(x[i][j]);
            }
        }
        """
        bar = """
        void foo(int **x, int n) {
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++)
                    print(x[a][b]);
            }
        }
        """

        with z3repr_options(integer_repr="int"):
            self.assertTrue(are_equivalent(foo, bar))
            self.assertFalse(are_equivalent(foo, bar.replace("++", "+=2")))

    def test_irrelevant_path_conditions_are_ignored(self):
        foo = """
        void foo(char *s, int n) {
            if (n > 0) {
                if (n == 1) if (s[0] == 'c') return;
                printf("%s", s);
            }
        }
        """

        bar = """
        void foo(char *s, int n) {
            if (n >= 1) {
                if (n == 1) if (s[0] == 'c') return;
                for (int i = 0; i < n; ++i);
                printf("%s", s);
            }
        }
        """

        baz = """
        void baz(char *s, int n) {
            if (n >= 1) {
                if (n == 1) if (s[0] == 'c') return;
                for (int i = 0; i < n; ++i) {
                    if (i == 4) break;
                }
                printf("%s", s);
            }
        }
        """

        self.assertTrue(are_equivalent(foo, bar))
        self.assertTrue(are_equivalent(foo, baz))

    def test_function_pointer_calls(self):
        applyarg = """void applyarg(int arg, int (*f)(int)) { f(arg); }"""
        applyx = """void applyx(int x, int (*g)(int)) { (*g)(x); }"""
        applynext = """void applynext(int x) { int (*f)(int) = getf(); f(x); }"""

        self.assertTrue(are_equivalent(applyarg, applyx))
        self.assertFalse(are_equivalent(applyx, applynext))

    def test_function_pointer_in_struct(self):
        applya = """struct s { int x; int (*f)(int); }; void applya(struct s *a) { a->f(a->x); }"""
        applyb = """struct t { int y; int (*g)(int); }; void applyb(struct t *b) { b->g(b->y); }"""
        applyc = """struct s { int (*f)(int); int x; }; void applyc(struct s *c) { c->f(c->x); }"""

        self.assertTrue(are_equivalent(applya, applyb))
        self.assertFalse(are_equivalent(applya, applyc))

    def test_differing_loop_conditions_results_in_nonequivalent_phis(self):
        ltn = """
        int ltn(int n) {
            int sum = 0;
            for (int i = 0; i < n; ++i) sum += i;
            return sum
        }
        """

        ltnm1 = ltn.replace("i < n", "i < n - 1")

        self.assertFalse(are_equivalent(ltn, ltnm1))

    def test_differing_break_statements_results_in_nonequivalent_phis(self):
        break4 = """
        int foo(int n) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += i;
                if (i == 4) break;
            }
            return sum
        }
        """
        break3 = break4.replace("4", "3")

        self.assertFalse(are_equivalent(break4, break3))

    def test_differently_named_but_structurally_identical_structs(self):
        # NOTE: right now the recursive definitions (of the lists) are not checked;
        # this test is written defensively for future recursive-equivalence support.
        account = """
        struct list { int val; struct list * next; };
        struct account { int id; char * name; struct list * transactions; };
        struct account * loadaccount(int id) {
            struct account * acct = malloc(sizeof(struct account));
            acct->id = id;
            acct->name = loadname(id);
            struct list * balance = malloc(sizeof(struct list));
            balance->val = loadbalance(id);
            balance->next = NULL;
            acct->transactions = balance;
            return acct;
        }
        """

        userinfo = """
        struct transaction { int amount; struct transaction * future; };
        struct userinfo { int uid; char * desc; struct transaction * ts; };
        struct userinfo * userinit(int uid) {
            struct userinfo * info = malloc(sizeof(struct userinfo));
            info->uid = uid;
            info->desc = loaddesc(uid);
            struct transaction * ts = malloc(sizeof(struct transaction));
            ts->amount = loadbalance(uid);
            ts->future = 0;
            info->ts = ts;
            return info;
        }
        """

        self.assertTrue(are_equivalent(account, userinfo))

    def test_dataflow_dependency_in_loop_head(self):
        i = """int decrement(int i) { while (i--); return i; }"""
        self.assertTrue(are_equivalent(i, i))
        

    # def test_returned_simple_heap_equivalence(self):
    #     decl = "char * malloc(long size);" # technically not how malloc is defined but it's fine for the purposes of the test.
    #     cxmalloc = decl + """char * cmalloc(int size) { char * ptr = malloc(size); return ptr; }"""
    #     stralloc = decl + """char * stralloc(int size) { char * ptr = malloc(size); ptr[size - 1] = 0; return ptr; }"""
    #     myalloc = decl + """char * myalloc(int len) { char * p = malloc(len); p[len - 1] = 0; return p; }"""

    #     self.assertTrue(are_equivalent(stralloc, myalloc))
    #     self.assertTrue(are_equivalent(cxmalloc, stralloc))
