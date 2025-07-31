# Artifical Intelligence in Games (Take Games as a Simulation for Reality)

## Characteristics of Games
### Deterministic vs Chance
- *Deterministic*: No chance element
### Perfect vs Imperfect Information
- No hidden element
    - In card games, other players cards are hidden making it imperfect
### Zero-Sum
- One player's gain is another player's loss

### Example
- Chess
    - 2 Players
    - No Chance Involved
    - Perfect Information
    - Zero-Sum Game

## Search Algorithm in Games
### Example
- Chess:
    - State - Board Configuration
    - Initial State - Initial Board Configuration
    - Terminal States - Checkmate States
    - Operators - Legal Chess Moves
    - Utility Function - Numeric Function Applied to Terminal States Determining
                         if Player X has Won or Loss
    - Evaluation Function - Numeric Function Applied to Non-Terminal States Determining
                            if Player X is in an Advantageos or Disadvantageous State
    - Game Tree - Tree showing all possible scenarios

## Evaluation Function

Similar to the heuristic functions, but tells us how favourable the current state is
for player X.

## Minimax Algorithm
Given two players MIN and MAX.

Assume that each player plays optimally, meaning that:
- MIN always picks the MINIMUM value <- This is the evaluation function for MIN
- MAX always picks the MAXIMUM value <- This is the evaluation function for MAX

In a search tree, because its a two player game, each layer alternates representing each turn:
```txt
                      A             MAX
                     / \
                    /   \
                   /     \
                  B       C         MIN
                 / \     / \
                /   \   /   \
               D     E F     G      MAX
```
So in the example above, let's start at `A`. 
- It's MAX's turn, so he would pick the maximum value between B and C.
- Say he picks B, and B > C.
- Now its MIN's turn. And MIN always picks the minumim value.
- So MIN picks the smallest value between D and E.

Now, assuming that each player moves optimally, there will always be 1 path throughout
the entire graph. However, to determine that path, we have to compute the evaluation values
starting from the leaf nodes.

So we assign the minimum value from D and E to B becuse if the board is in B state, then
MIN wil always pick the smallest value. 

Similarly, we assign the minimum values from F and G to C.

Now we compare B and C and assign the maximum value to A, because MAX will always pick the
maximum value between B and C.

>[!note] NOTE
> MIN will always play the minimal value because its better for MIN. If MIN does not
> play the minimal value, then MAX will play even better/get a better result.

```txt
                  A (3)             MAX
                 /|\
                / | \
               /  |  \
              /   |   \
             /    |    \ 
         B(3)    C(2)   D(2)        MIN
        //|      /|\      |\\
      / / |      ...       ...
    /  /  |
E(3) F(9) G(8)
```
## Resource Limits/Alpha-Beta Pruning
For larger games, we definitely cant compute all states. As such we con only look ahead
a specific amount of steps.

Or instead of constraining the amount we can look forward. We can narrow our vision
by determining sub-trees that are not worth expanding.

In the example above. If max will always pick the highest number, then we clearly do not
need to look at the sub-trees of C and D. As such we "prune" those branches.

Now, the most obvious thought is, to get those values, don't we need to look from the leaf
nodes? Making "pruning" essentially pointless extra steps?

> [!note] NOTE
> It's like seeing 2 moves in advance!
> You can only really prune with 3 layers, it's possible with 2 but less accurate 

Technically yes, but here is an example where it is less nodes to generate:
```txt
         >=3        MAX
          /|
         / |
        /  |
       /   |
      /    |
     /     |
    /      |
   3     <=2        MIN
  /|\     /|\
 / | \   / | \
3  12 8 2  X  X     
```
So if we look here, to get 3, we iterate over all the children and determine that 3 
is the smallest child. Similarley to the neightbouring branch, its first child is 2.

Now if we let `2` become the "parent" the MAX player would never pick this, due to 3 
being bigger. So it doen't matter what in the other two `X` because even if its a 
bigger number MIN would pick `2` and MAX would NEVER pick `2` over 3.

So why bother generating the other two nodes if we are never going to pick the node 
labelled `2`.

## Still Not Enough

Even though we can prune branches, this is till ALOT of data. 

If we wanted to, we could always do a search for a limited time, and return the deepest 
search found. This would drastically reduce resources needed, but may not give us the
best result.

Other attempts have included modifying the evaluation function to include weights/biases
to promote searching branches of higher likelihood of occurance.

## Horizon Effect
In some cases, it may seem that Player X has an advantege, however, things can change
in 1 move. For example, in certain states of a chess board, black may have more pieces,
but white could get a queen due to pawn. And the program would not be able to catch that
if it only evaluates piece value per state.

Solutions to this include:
- Secondary search to ensure there are no additional pitfalls
- Evaluation function takes this into account
- Iterative Deepening Search

## Games of Chance

In the case of chance based games, we add weights to each edge, representing the 
probabilities of each state occuring. In the worst case, we add chance nodes as we need to
account for potential combinations of dice rolls and coin flips.

In cases like this, EXACT values do matter.







