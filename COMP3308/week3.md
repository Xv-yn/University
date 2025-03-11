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

A dominant heuristic is basically just a larger heuristic value that is
also admissable.

h_2(n) > h(n)

1. In A*, ∀ n with f(n) < f* will be expanded
   Translation: for all nodes, where `estimated_cost` < `true_cost`,
                will be expanded
2. By rearranging the formula f(n) = g(n) + h(n) into h(n) = f(n) - g(n)
   and because f(n) < f* and substitution, we can say:
   h(n) < f* - g(n)
3. We know that h_2(n) > h(n) and that h\_2(n) is admissable
   This tells us that h\_2(n) is more informed than h(n).
   Because there will be fewer nodes satisfying h\_2(n) than h(n) < f∗ − g(n)
4. Therefore, all nodes expanded by A* using h\_2(n) will also be expanded 
   by h(n). BUT, h(n) may xpand otehr nodes as well.

We prefer to use dominant heuristic values because they expand less nodes
and hence, gives an output faster.

## Creating/Determining/Inventing Admissible Heuristics (creating the heuristic function)
To do this, we form a relaxed version of the problem.

For example, if we use the Manhattan Distance we get the "bee line" solution
for each node. However, this does not account for other potential 
rules/conditions.

For Example:
```txt
Start State                 End State
+---+---+---+               +---+---+---+
| 5 | 2 | 8 |               | 1 | 2 | 3 |
+---+---+---+               +---+---+---+
| 3 |   | 6 |               | 4 | 5 | 6 |
+---+---+---+               +---+---+---+
| 7 | 4 | 1 |               | 7 | 8 |   |
+---+---+---+               +---+---+---+
```
The manhattan distance for `1`,(up, up, left, left), in the above problem 
is 4. But it does not take 4 moves to get 1 into the top left position as
the manhattan distance does not account for the fact that you can only move
a tile to the blank space.

Instead of using the manhattan distance, we can use another method, where 
any tile can be swapped/moved anywhere. As a result, we could move `1` to 
the correct position in 1 move.

These are valid methods to determine the heuristic value, but may not be 
the best methods. This is because (by intuition) these methods are severely
optimistic and are less than the actual number of moves, as it definitely 
does not take 1 or 4 moves to move 1 tile to the correct position.

To formally construct a relaxed problem, we define the problem in formal 
language, using the above example:
- A tile can move from Square A to Square B if 
- (CONDITION) A is adjacent to B and B is blank

We can generate 3 relaxed problems by removing 1 or both conditions:
- Relaxed Problem 1
    - A tile can move from Square A to Square B if 
        - A is adjacent to B
- Relaxed Problem 2
    - A tile can move from Square A to Square B if
        - B is blank
- Relaxed Problem 3
    - A tile can move from Square A to Square B 



## Terminology
 - Manhattan Distance - the sum of distance from the start position 
   to the goal position for each node.

