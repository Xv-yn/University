# Key Dates

- [ ] (10%) Weekly Quiz 
        Individual
     Est. Given: Every Thursday
            Due: Every Wednesday

- [ ] (7.5%) Assignment 1 
        Individual
     Est. Given: 10 Mar 2025 Week 3 Monday
            Due: 26 Mar 2025 at 23:59 Week 5 Wednesday

- [ ] (7.5%) Assignment 2 
        Individual
     Est. Given: 26 Mar 2025 Week 5 Wednesday
            Due: 09 Apr 2025 at 23:59 Week 7 Wednesday

- [ ] (7.5%) Assignment 3 
        Individual
     Est. Given: 23 Apr 2025 Week 8 Wednesday
            Due: 07 May 2025 at 23:59 Week 10 Wednesday

- [ ] (7.5%) Assignment 4 
        Individual
     Est. Given: 07 May 2025 Week 10 Wednesday
            Due: 21 May 2025 at 23:59 Week 12 Wednesday

- [ ] (60%) Final Exam 
        Individual
     Est. Given: Formal exam period 
            Due: Formal exam period

# Time Complexity Formal Definition

## Big O-Notation

Say we have some function f(n) is O(g(n)), then:

> f(n) ≤ c ⋅ g(n) for all n ≥ n_0
    
This basically means that at some point, a large enough n_0, f(n) 
is at most some multiple of g(n).

Example:
    1. Let f(n) = 5n^2 + 3n + 7
    2. Let g(n) = n^2
    3. We want to show that f(n) is O(n^2)
    4. By "guessing", we choose 
        c = 6 and 
        1 n_0 = 1, 
        we can see that for all n ≥ 1.
                     f(n) ≤ c ⋅ g(n)
            5n^2 + 3n + 7 ≤ 6 ⋅ n^2
            

## Big Omega-Notation

A function f(n) is Ω(g(n)) if there exists a constant c > 0 and 
a threshold n_0 such that for all sufficiently large n ≥ n\_0:

f(n) ≥ c ⋅ g(n) f(n)≥c⋅g(n)

If f(n) = 3n^2 + 5n + 7 and g(n) = n^2, then f(n) is Ω(n^2) 
because f(n) is always greater than some constant multiple 
of n^2, like 2n^2.

## Big Theta-Notation

A function is Θ(g(n)) if it is both O(g(n)) and Ω(g(n)).

Mathematically, there exist two positive constants c_1 and c\_2 such 
that for sufficiently large n:

c_1 ⋅ g(n) ≤ f(n) ≤ c\_2 ⋅ g(n)

If a function is O(n^3) and Ω(n^2). Then there is no defined Θ(g(n)).

## Growth Functions

Slowest                                         Fastest
Exponential (n^k) < Polynomial (n^c) < Logarithm (log n)


