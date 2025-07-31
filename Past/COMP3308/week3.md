# A* Search

- Evaluation Function
    - Uses the formula f(n) = g(n) + h(n)
                            path_cost + heuristic_value
    - To handle ties, the first node added to the queue is chosen first

f(n) is also called the f-value. This is basically an estimate of the distance of the 
entire path.

Using an example, say I'm at the start of a walking path. On estimate I need to walk 10 
kilometers to reach the end. 
In this scenario, the path cost = 0 and the heuristic value = 10, thus, the f-value = 10
- path cost = 0 because I haven't walked any distance yet
- heuristic value = 10 because I think I need to walk 10 km
- f-value = 10, which the total path distance I need to walk from start to end

Now that I'm halfway, I have already walked 5 kilometers. Looking at the estimate I 
need to walk another 5 kilometers to reach the end.
In this scenario, the path cost = 5 and the heuristic value = 5, thus, the f-value = 10
- path cost = 5 because I have already walked 5 kilometers
- heuristic value = 5 because I think I need to walk another 5 km to reach the end
- f-value = 10, which is still the total path distance I need to walk from start to end

## Admissible Heuristics
Each node in a graph/tree is given a heuristic value (calculated by 
the evaluation function). 

An admissible heuristic is a number that is given to a node that guesses its distance
from the goal. Note that this value is optimistic, like travelling through a city with
no traffic.

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

## Heuristic Consistency

```txt
             S                                S
            / \                              / \
           /   \                            /   \
          /     \                          /     \
  c(S,n) /       \ c(S,n`)              2 /       \ 2
        /         \                      /         \
       /           \                    /           \
      /   c(n,n`)   \                  /      2      \
h(n) n ------------- n` h(n`)     [3] a ------------- b [4]
      \             /                  \             /
       \           /                    \           /
        \         /                      \         /
  c(n,G) \       / c(n`,G)              4 \       / 5
          \     /                          \     /
           \   /                            \   /
            \ /                              \ /
             G                                G 
```

Heuristics consistency is important because this calculation prevents revisiting 
an already visited node.

To determine if a heuristic is consistent, the h(n) must meet the following equation
for ALL connected edges (even the one it came from!): 

h(n) <= c(n, n\`) + h(n\`)

> [!important]
> This equation basically says:
> The distance to the goal from where you are now (h(n)),
> must be less than or equal to (<=)
> the cost of travelling to the neighbouring node (c(n, n\`))
> and the distance from the neighbouring node to the goal combined (+ h(n\`))

> [!caution] Intuititively
> Assuming that all the heuristics that we check are admissable (meaning that they are
> optimistic) we are saying that the estimated distance to reach the goal from where we 
> are must always be less than or equal to the cost to travel to any neightbour plus 
> their estimate to the goal.

```txt
    We need to travel 3 km to reach (G)
              |
              v
[2]          [3]          [1]          [0] 
(S) -------> (A) -------> (B) -------> (G)
       3            4             3
                 '----------' 
                       ^
                       |
    We need to travel 4 + 1 = 5 km to reach (G)

Putting this together, we get "optimistically we need to travel 3 km, if we go to node 
B we need to travel a total of 5 km"
```

In simpler terms, lets say we were at node `a` in the above graph. It would make 
more sense to directly travel to `G` with a cost of 4 instead of going to `b` then `G`.
Consistency is measuring whether or not it is logical to take that "detour".

> [!note]
> A consistent heuristic is ALWAYS admissble. 
> BUT an admissble heuristic is NOT always consistent.

# Optimization Problems

Up until now, the goal was to find a path to a certain state.
Now, we just want to find a certain state, regardless of path.

We can't use BFS, DFS, UCS, IDS, Greedy or A* due to how expensive it is
computationally.

So to do this, using each state, we can calculate a specific value, 
representing how optimal the current state is. This function is called:
"heuristic evaluation function".

Generalized Steps:
1. Find the global maximum (to get a feel for the upper bound of the dataset)
2. Find the global minimum (to get a feel for the lower bound of the dataset)
3. Start local search
    - Complete local search finds a valid goal state if it exists
    - Optimal local serach finds the best possible state associated with the
      global max or min

## Hill-Climbing Algorithm

This algorithm only works on a "small scale", more sepcifically, only a 
local area. If this algorithm cannot find a suitable state, then it can:
    - restart with a randomly chosen new starting state

### Potential Modifications
- keep track of states that have been visited
- check the neighbours neighbours instead of neighbours
- check multiple states ahead instead of only looking at neighbours
- occasionallly accept 2nd best moves (or worse)
- beam search, select the best 2 children of the next layer and so on

### Simulated Annealing
This method is basically, start at a certian node, with a "high temperature"
variable. This "high temperature" causes the algorithm to explore widely and
accept wrose moves often, after every move, the "high temperature" decreases,
meaning that the algorithm becomes more selective picking less worse moves 
until a goal is found.

### Descent
1. Starting at an intial state s
2. Find the best neighbouring state (the one with the lowest v(n) value)
3. Compare v(n) against the the current states v(s)
4. If v(n) is better than v(s) move to this state, else stop

### Ascent

1. Starting at an intial state s
2. Find the best neighbouring state (the one with the highest v(n) value)
3. Compare v(n) against the the current states v(s)
4. If v(n) is better than v(s) move to this state, else stop

## Genetic Algorithm

Literally DNA chromosome mixing/crossing over stuff.

1. Create a random population
    - Each individual is represented as a chromosome 
      (bitstring, list, or encoded parameters)
2. Apply a fitness function (a function that determines how good a 
                             solution is)
3. Select parents
    - Can be selected randomly, or based on fitness function
4. Crossover
    - Cut two/three/etc. segments and attach to opposite
    - Could also mix them up randomly
5. Mutation
    - Randomly change some of the "offspring"
    - E.g. Always flip the second bit like "0110" -> "0010"
6. Replacement
    - Replace the old population with the new ones and keep the best
7. Repeat until satisfactory solution is found 

# Terminology
 - Manhattan Distance - the sum of distance from the start position 
   to the goal position for each node.

