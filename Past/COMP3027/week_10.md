# NP and Intractibility

## Early NP-Complete Problems

1. PYSAT:
    - Input: Python program A and time bound t (unary).
    - Question: Is there input that makes A output “yes” in ≤ t steps?
    - Certificate: Input string.

    NP-complete via encoding any NP verifier as a Python program.

2. TMSAT:
    - Input: Turing machine M and time t.
    - Goal: Is there input c such that M accepts in ≤ t steps?
    - Also NP-complete by compiling Python programs into TMs.

## Cook Levin Theorem

> The Cook Levin Theorem States:
> The Boolean satisfiability problem (SAT) is NP-complete.

In simpler terms, this means:
- SAT is in NP: Given a Boolean formula, we can check whether a proposed 
  solution (a variable assignment) satisfies it in polynomial time.
- SAT is NP-hard: Any problem in the class NP can be transformed into an 
  instance of SAT in polynomial time.

## Circuit Tree

CIRCUIT-SAT: Is there an input assignment that makes a Boolean circuit 
             output 1?

Cook-Levin Theorem: CIRCUIT-SAT is NP-complete.
- Proof idea: Reduce TMSAT to CIRCUIT-SAT using simulation.

## Other Reductions

3-SAT:
- Shown NP-complete via reduction from CIRCUIT-SAT.
- Used to reduce to many other problems.

Graph Coloring:
- k-COLOR: Proper coloring with ≤ k colors.
- 3-COLOR is NP-complete via reduction from 3-SAT using:
    - Truth gadgets (T, F, O)
    - Variable gadgets
    - Clause gadgets (ensure at least one literal is true)

Numerical Problem: 
- Subset Sum 
    - Given set S={v1,...,vn} and 
    - target t
    - is there a subset summing to t? 
    - Vertex Cover ≤ Subset Sum via encoding of vertices and edges using 
      digit positions

## Hamiltonian Cycle

Undirected: Is there a simple cycle covering all vertices?

In simpler terms, given a weighted or unweighted graph, is there a way to 
traverse the graph such that:
- All nodes are visited
- No nodes are visited more than once

Directed version (DIR-HAM-CYCLE): Uses gadgets to simulate direction.

3-SAT ≤ DIR-HAM-CYCLE:
- Build graph where:
    - Paths encode variable assignments.
    - Clause nodes added to link satisfying assignments.

## Travelling Salesman Problem (TSP)

TSP: Is there a tour of length ≤ D through all cities?

In simpler terms, given distances (or costs) between cities, what is the 
shortest possible Hamiltonian cycle?

HAM-CYCLE ≤ TSP: Encode graph as city distances.

> [!note] Reminder
> HAM-CYCLE ≤ TSP
> HAM-CYCLE is reduced to TSP


