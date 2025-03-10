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

* Expanded Nodes - These nodes have already been processed and placed into
                   the 'visited' list
* Frontier Nodes - These nodes are on the 'temp_list'
* Evaluation Function - This function is what we use to traverse the tree.
                        It is how we decide which node to visit next.
* Heuristic Function - This function assigns heuristic values to each node.

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

## Breadth First Search (BFS)
```txt
visited_nodes = []
temp_queue = []

append Start_Node into temp_queue

while temp_queue is not empty:
    node <- dequeue(temp_queue)
    for child in node.children:
        if child not in visited_nodes
            enqueue(child, temp_queue)
    append node to visited_nodes
```
Time Complexity: O(b^d)

## Depth First Search
```txt
visited_nodes = []
temp_stack = []

append Start_Node into temp_stack

while temp_stack is not empty:
    node <- pop(node, temp_stack)
    if node is not in visited_nodes:
        append(node, visited_nodes)
        for child in reverse(node.children)
            append(child, temp_stack)    
```
Time Complexity: O(b^d)

## Uniform Cost Search
```txt
# Assumed node structure Node(cost, children[], other_data)

visited_nodes = []
priority_queue = []

append (0,Start_Node) into priority_queue

while priority_queue is not empty:
    cost, node <- priority_dequeue(priority_queue)
    if node in visited_nodes:
        append (cost, node) into visited_nodes
        for child_cost, child in node.children:
            enqueue((child_cost + cost, child), priority_queue)
```
Time Complexity: O(b^(C∗/eˉ)d)

## Iterative Depth Search
```txt
def IterativeDeepeningSearch(root, list)
    for depth from 0 to ∞ :
        result ← DepthLimitedSearch(root, list, depth)
        IF result ≠ "Not Found" THEN
            RETURN result
    return list 

def DepthLimitedSearch(node, list, limit)
    list.append(node)
    
    if limit = 0 then
        return   

    for child in node.children
        result ← DepthLimitedSearch(child, list, limit - 1)
    return
```
Time Complexity: O(b^d)


