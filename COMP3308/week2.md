# Searching

When searching a graph, we can "reshape/reorganize" the graph to be viewed
as a tree.

Imagine a tree:
- Each Node in the tree is NOT a `state`
    - A node HOLDS the state AND knows it parents and children
- Each branch/edge in the tree is NOT an `operator`
    - An Edge HOLDS the operator but also other details like weights
- Some branches/edges are weighted in the tree, called `cost`

For a search, there must be:
- `Starting State` (starting vertex/node)
- `Goal/End State` (ending vertex/node) | Can have more than 1 goal state
- `Operators` (set of valid edges/paths can be taken from each state)
- `Path Cost Function` (Assigns a numerical value to each path)

The `Solution` is a path from the starting state to the goal state
- `Optimal Solution` is the path with the lowest path cost

`State Space` is the set of all possible states that can be reached 
in the algorithm.

# Abstraction
The simplification of a problem by removing unnecessary problems is called
`Abstraction`.

For example, when finding a path from point A to point B on a map, the 
weather or one's blood type is irrelevant.

# Searching for a Solution

1. We generate the possible moves in the form of a tree
2. We run a search algorithm (DFS, BFS) to find the goal state

# Search Strategies
Evaluation Criteria:
- Completeness
    - The algorithm is able to ALWAYS find a solution (if the solution exists)
- Optimality
    - The algorithm is able to ALWAYS find the least cost path
- Time Complexity
    - Time taken to find the solution (measured in the number of nodes generated)
- Space Complexity
    - Maximum number of nodes in memory

> [!important] Terminology
> - `b` - max branching factor (maximum number of branches of a node in a tree)
> - `d` - depth of optimal solution 
> - `m` - maximum depth of space state (how far down the tree goes, can be infinite)







