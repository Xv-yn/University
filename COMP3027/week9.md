# Reductions, NP and Intractibility

## 3 Main Reduction Methods

1. Reduction by simple equivalence. (VC and IS)
    - Vertex Cover (VC)
        - Set of vertices such that every edge in the graph touches at 
          least one vertex in the set    
    - Independent Set (IS)
        - A Vertex Cover where no two vertices in the set are connected by 
          an edge
    - Reduction Example:
        - Given the following graph:
        ```txt
        A --- B --- C
        ```
        - The Vertex Cover of this graph would only consist of 1 node, `{B}`
        - The Independent Set problem can be reduced to the "inverse" of the 
          Vertex Cover Problem, where it is equal to all nodes except for the 
          nodes in the VC problem, `{A,C}`. 
2. Reduction from special case to general case.
    - Vertex Cover problem can be reduced to Set Cover problem
    -   special case       can be reduced to    general case  
    - Set Cover = General Case
        - This is because the Set Cover problem takes in:
            - A universe U (any kind of elements)
            - A collection of subsets S₁, S₂, ..., Sₘ
             - Must choose k of these subsets such that their union covers U
        - This is very flexible and abstract
    - Vertex Cover = Special Case
        - Each subset is tied to a graph structure — it's not arbitrary
3. Reduction by encoding with gadgets.


## Definition of P

Problems solvable in polynomial time.

## Definition of NP

Problems where YES-instances can be verified in poly-time (existence of a 
certificate/witness). 

Example: 

Where’s Waldo – verifying Waldo's location is easier than finding him.

## NP Completeness

Problems that are in NP and every problem in NP reduces to them.

If any NP-complete problem is solvable in poly-time → P = NP.

In other words, if I can solve problem X (assuming X is NP-complete) in 
polynomial time, then this NP problem can be solved in P time (P = NP). 

## Cook Levin Theorem


