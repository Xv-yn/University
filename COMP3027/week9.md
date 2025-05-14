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
    - The use of "gadgets" generally means we use an circuit/graph 
      representation
    - Instead of direct reduction to a graph, we build the graph/circuit
    - 3SAT
        - The 3SAT problem is a CNF formula where each clause contains exactly 
          3 literals and can we assign True/False values to variables so that
          it outputs to True
        - e.g. 
          (¬x₁ ∨ x₂∨ x₃) ∧ (x₁ ∨ ¬x₂ ∨ x₃) ∧ (x₁ ∨ x₂ ∨ x₃) ∧ (¬x₁ ∨ ¬x₂ ∨ ¬x₃)
        - We can reduce this problem to the Independent Set problem
        - We construct a graph such that we form 4 triangles (1 trangle for each
          clause)
          ```txt
          (¬x₁ ∨ x₂∨ x₃)

               ¬x₁
              /   \
             x₂----x₃

          ```
        - Because it is in CNF, if ¬x₁= True then it doesn't matter if x₂ or x₃ 
          are True or False, the output of the clause (¬x₁ ∨ x₂∨ x₃) is True.
        - The same could be said if x₂= True instead. Where it doesn't matter if 
          ¬x₁ or x₃ are True or False, as the output of the clause (¬x₁ ∨ x₂∨ x₃) 
          would still be True.
        - Now when we transform this into the graph and apply the Independent Set 
          Problem, the above thought process can be applied. 
            - Let's say we select ¬x₁ in the Independent Set, we cannot select 
              the x₂ or x₃ vertex (because they are neighbouring vertices).
            - We could choose x₂ instead for the Independent Set, but then we 
              cannot select ¬x₁ or x₃ (because they are neighbouring vertices)
        - Now this only applies to 1 clause, but to connect the clauses we also 
          connect literals to each of its negations.
          ```txt
          (¬x₁ ∨ x₂∨ x₃) ∧ (x₁ ∨ ¬x₂ ∨ x₃)

               ¬x₁ ---------- x₁
              /   \         /   \
             x₂--- x₃     ¬x₂--- x₃
               \__________/
          ```
        - Now say we apply the above thinking to TWO clauses, lets say we pick
          x₂ in the Independent Set (picking this basically means that x₂ is 
          True). This edge connecting the literal to its negation ensures that we
          CANNOT pick ¬x₂ in the second clause because ¬x₂ MUST be false.
        - This is why we connect each literal to each of its negation.

## NP-Hard Problems

## NP-Complete Problems


