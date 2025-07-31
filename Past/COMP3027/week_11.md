# How to handle NP-Complete Problems

Since polynomial-time solutions for NP-complete problems are unlikely, we 
must sacrifice at least one of:
- Optimality
- Generality
- Efficiency

Techniques:
- Approximation Algorithms (fast, non-optimal)
- Heuristics (practical but no guarantees)
- Exact Exponential-Time Algorithms
- Integer Linear Programming
- Solving restricted instances (e.g., trees, small solution size)
- Parameterized Algorithms (efficient for small parameter values)

## Independent Set on Trees

Solvable in O(n) time using a greedy or DP algorithm 

For weighted trees, use dynamic programming: 
- Track max weight with/without each node 
- Postorder traversal ensures efficiency 

## Small Vertex Cover

If you want to cover an edge (u, v) in a graph, then:
- Either u is in the vertex cover, or
- v is in the vertex cover, or
- both are (but you never need both — one suffices).

If G has a vertex cover of size ≤ k, then some vertex in the cover must be 
either u or v (to cover edge (u, v)).

So removing one of them and adjusting the rest gives you a smaller problem 
on a smaller graph.

If the solution size k is small, a brute-force method is feasible

Runtime improved from O(n^k) to O(2^k x kn).
- Uses branching on edges and recursion

## Approximation Algorithms

Find the smallest Vertex Cover:
- 2-approximation using:
    - Greedy edge-picking strategy
        - Start with an empty cover set C = ∅ 
        - While there are edges left in E: 
            - Pick any edge (u,v) ∈ E 
            - Add both u and v to the cover: C = C ∪ {u,v} 
            - Delete all edges next to u or v 
            - Answer = len(C)/2

    - Matching-based proofs via weak duality

Weighted Vertex Cover:
- Pricing method achieves 2-approximation using edge payments and fairness 
  constraints

Load Balancing (Identical Machines):

- Given 3 machines, and X jobs, how to assign jobs?

- List Scheduling: Assign job to least-loaded machine
    - Proven 2-approximation (Graham’s algorithm)
- LPT Rule (Longest Processing Time first):
    - Improves to 3/2 or even 4/3 approximation

## Randomized Algorithms

MAX-3SAT:
- Satisfy as many clauses as possible (NP-hard)
- Random assignment satisfies ~7/8 of clauses on expectation
- Leads to Johnson’s algorithm: Random sampling until ≥ 7k/8 clauses are 
  satisfied

In simpler terms:
- For a single clause with 3 literals: 
    - The only way it’s not satisfied is if all 3 literals are false 
    - That happens with probability: (1/2)^3 = 1/8
    - So the chance a clause is satisfied is: 1 − 1/8 = 7/8
    
If you have m clauses, the expected number satisfied is: 

7/8 x m 

Therefore, by the law of expectation, there must exist some assignment that 
satisfies at least 7/8 of the clauses.


