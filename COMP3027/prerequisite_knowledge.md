# Running Times

- What is an algorithms worst case running time?
    > An algorithms worst case running time is the maximum time an algorithm takes to 
    > complete given the worst possible input size of n. It is denoted using Big-O notation.

- What does it mean when we say "An algorithm runs in polynomial time"? 
    > This means that the time taken for an algorithm to be completed can be expressed in 
    > a polynomial function given an input of size n. Specifically, O(n^k), where k is 
    > some constant.

- What does it mean when we say "An algorithm is efficient"?
    > An efficient algorithm can:
    >   - handle larger inputs without significant increase in resource consumption
    >   - make optimal use of avaliable resources, balancing space and time complexity
    >   - handle different types of input gracefully and maintain performance

## Asymptotic Analysis

- Big-O Notation (O(·))
    > Represents the **UPPER** bound of an algorithms running time.
        > Sorting Algorithm Input Example:
        > Given an already sorted list

- Big-Theta Notation (Θ(·))
    > Represents the **AVERAGE** bound of an algorithms running time.
        >Typical performance for random inputs

- Big-Omega Notation (Ω(·))
    > Represents the **LOWER** bound of an algorithms running time.
        > Sorting Algorithm Input Example:
        > Given a list sorted in a way that causes the algorithm to 
        > perform the maximum number of comparisons and swaps, thereby
        > triggering every iteration.

### Data Structures

- Linked Lists
    > A list where each node contains a pointer to the next (sometimes also the previous)

- Queues
    > First in, First out

- Stacks
    > First in, Last out

- Balanced Binary Trees
    > Binary tree where height of left and right branches are the same

#### Graphs

- Notation G(V,E)
    > G indicates that it is a graph
    > V indicates the vertices
    > E indicates the edges

- Adjacency List
    > * Each vertex has a list of all vetices it is attached to
    > Example:
    > ``` txt
    > A: B, C
    > B: A, D
    > C: A, D
    > D: B, C
    > ```
    > Pros:
    > * Space efficent
    > * Faster iteration over edges
    > 
    > Cons:
    > * Edge lookup time is O(n)

- Adjacency Matrix
    > * Each cell has a 0 or 1 indicating if there exists an edge between two vertices
    > Example:
    > ``` txt
    >     A B C D
    > A [ 0 1 1 0 ]
    > B [ 1 0 0 1 ]
    > C [ 1 0 0 1 ]
    > D [ 0 1 1 0 ]
    > ```
    > Pros:
    > * Edge lookup time is O(1)
    > * Simplicity
    > 
    > Cons:
    > * Space inefficient
    > * Iteration over edges is slower

* Simple Paths
    > A path in a graph that does not repeat any vertices

* Cycles
    > A path in a graph that:
    >   * Begins and ends on the same vertex
    >   * No other vertices are repeated except the start and end

* Trees
    > A type of graph with the following characteristics:
    > * Does not contain any cycles
    > * Every pair of vertices in a tree is connected by exactly one path
    > * If there are n vertices, then there must be n - 1 edges 

* Unrooted Trees
    > A tree with an undesignated root

* Rooted Trees
    > A tree with a designated root
    > 
    > There must be a statement of some sort or label on the graph
    > specifying the designated root

* Bipartite Graph
    > Two disjointed sets of vertices and each edge connects a vertex in 
    > one set to the other set.
    > There are no edges between vertices within the same set
    > 
    > There are no Odd-Length cycles

##### Graph Traversal


