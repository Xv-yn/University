# A* Search

- Evaluation Function
    - Uses the formula f(n) = g(n) + h(n)
                            path_cost + heuristic_value
    - To handle ties, the first node added to the queue is chosen first

## Admissible Heuristics
Each node in a graph/tree is given a heuristic value (calculated by 
the evaluation function). 

To determine if a heuristic is admissible, the heuristic value must be 
less than or equal to the true cost to readch the goal.

h(n) <= h*(n)
heuristic value <= total cost to goal node

For example:
```txt
            A [3]
           / \
        4 /   \ 3
         /     \
        B [6]   C [3]
       / \       \
   11 /   \ 2     \ 1
     /     \       \
   (D)[0]   E [0]   (F)[0]
```

`A [3]` is an admissible heuristic because 3 <= 4.
`C [3]` is NOT an admissible heuristic becase 3 <= 1.

Admissible heuristics are optimistic, meaning that they think the cost
of solving the problem is less than it actually is.

If a graph contains an admissble heuristic, then A* may not be complete
or optimal.

In simpler terms, A* may not find a solution or the solution it finds 
may not be the best solution.

## Dominant Heuristics


## Terminology
 - Manhattan Distance - the sum of distance from the start position 
   to the goal position for each node.

