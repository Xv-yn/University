# Dynamic Programming

The essence of dynamic programming is optimization of an algorithm.

In simpler terms, it means that 'dynamic programming' is another approach
to solve a problem in a more efficient manner.

Dynamic Programming is very un-intuitive. Instead of solving the problem
from the ground up, we solve it from the top down.

The essence of dynamic programming is turning the problem into smaller
blocks and stacking them up into a tower that is the solution. (Using
solutions to a smaller part of the problem and building them into the
actual solution)

# 1 Dimensional Dynamic Programming

Starting with the simplest form of dynamic programming, a 1 dimensional 
problem.

Say I have only 1c, 7c and 10c coins. Given some number n, how can I make an
algorithm that will give a value n made of the least amount of coins?

To solve this we need to assume we have a solution already. Like so:

## Step 1

Let OPT(i) be the minimum number of coins needed to get a value of i.
For example, OPT(14) is 2, because I only need 2x 7c coins.

Formally:

OPT(i) = optimal solution where the minimal number of coins sum to i.

> [!note] NOTE
> This kind of definition is very flexible but MUST be worded properly.

## Step 2

Now, we need to find a way to build up to that solution.

To do this, we need to define the possible cases from the optimal
solution. In this case, we "build down" from the existing assumed
solution/we find the previous iteration. 

- Case 1: Assume there exists an optimal solution for i - 1. For example,
  if i = 14, then we would assume there is a solution for 13.
    - We choose i - 1, specifically the "1", because 1c is one of the coins
      we can use.
    - Now because OPT(i) is a number (the minimal number of coins), we 
      can represent this like so:
      - OPT(i) = OPT(i-1) + 1
      - We choose the "+ 1" here because we add a coin to the optimal solution
      - Its like saying the optimal solution for 14 is equal to the optimal
        solution for 13 plus 1x 1c coin.
        - OPT(14) = OPT(13) + 1

- Case 2: Same as above but for i - 7. For example, if i = 14, then we would
  assume there is a solution for 7 (14 - 7 = 7).
    - We choose i - 7 beecause 7c is one of the coins we can use.
    - Similarly, we represent this like so:
        - OPT(i) = OPT(i-7) + 1
        - This is like saying the optimal solution for 14 is equal to the optimal
          solution for 7 plus 1x 7c coin
          - OPT(14) = OPT(7) + 1 

- Case 3: Same as above but for 10.

Now putting this together, we now have 3 OPTimal solutions for OPT(i):
- OPT(i) = OPT(i-1) + 1
- OPT(i) = OPT(i-7) + 1
- OPT(i) = OPT(i-10) + 1

The problem now is how do we decide which case is the best?

In this case, we want the MINIMUM number of coins so we take the minimum of all
solutions. Like so:

OPT(i) = min(OPT(i-1) + 1, OPT(i-7) + 1, OPT(i-10) + 1)

Using 14 as an example, we get the following:

OPT(14) = min(OPT(14-1) + 1, OPT(14-7) + 1, OPT(14-10) + 1)
        = min(OPT(13) + 1, OPT(7) + 1, OPT(4) + 1)

Ok, so assuming that OPT(x) will give the optimal solution, we can say the above is correct.
This is using the previous solutions to build the current solution.

However, this must start at somewhere. Specifically the base cases. The bottom of the
solution.

## Step 3

This is creating the foundation on which the above reasoning is built.

We now have 10 base cases to build up from:
- OPT(0) = 0
- OPT(1) = 1
- OPT(2) = 2
- OPT(3) = 3
- OPT(4) = 4
- OPT(5) = 5
- OPT(6) = 6 
- OPT(7) = 1
- OPT(8) = 2
- OPT(9) = 3
- OPT(10) = 1

Using these base cases and substitution via recursion, we get the optimal solution.

> [!note] NOTE
> Note that this runs in O(n) time.

## Pseudocode

```txt
def OPT(i):
    if i == 0:
        return 0
    if i == 1:
        return 1
    if i == 2:
        return 2
    if i == 3:
        return 3
    if i == 4:
        return 4
    if i == 5:
        return 5
    if i == 6:
        return 6
    if i == 7:
        return 1
    if i == 8:
        return 2
    if i == 9:
        return 3
    if i == 10:
        return 1
    if i < 0:
        return float('inf')  # using infinity for invalid cases
    
    return min(OPT(i - 1), OPT(i - 7), OPT(i - 10)) + 1
```

## Another Example (Less Explanation)

Weighted Interval Scheduling - Assume jobs are sorted by finishing time

OPT(i) = the optimal solution of jobs from 1,2,...,i

```txt
OPT(i) =  max(
            OPT(j) + w_i # let j be the previous job with no overlap
            OPT(k)       # let k be the (i - 1)th job, aka the job right before i
          )
```
Base Case: OPT(0) = 0

# 2 Dimensional Dynamic Programming

This is similar to 1 Dimension, but instead it requires O(n^2) time.

The method is the same where:
1. Assume you have a solution
2. Build up the solution by determining cases
3. Defining the base cases

An example would be as follows:

## Example: Longest Common String

Given the strings `AABCXYZ` and `AXYZABC`, we want to find the longest common
subsequence inside both of these strins.

Now, from us just looking at it, we know that the Longest Common Subsequence
is `AABC`.

Now Step by step:

### Step 1

OPT(i)(j) = length of the longest common subsequence of first string (with 
            length i) and second string (with length j)

So using the above example of strings `AABCXYZ` and `AXYZABC`, where `i = 7`
and `j = 7`,we assume that OPT(7)(7) = 4, because the LCS is `AABC`.

### Step 2

- Case 1: If first_string[i] != second_string[j]
    - This case is basically saying if a specific character on both
      strings are not the same.
    - Using the above example, if `i = 6` and `j = 6`. Where the highlighted 
      character is denoted using `| |`
      `AABCXY |Z|`
      `AXYZAB |C|`
    - But you see, at this point we've identified the case, but not what we
      should do once we've identified it.
    - This part is similar to 1 Dimension, but this time we are iterating over
      two strings instead of 1. Hence we take the optimal solution from right
      before, which is either:
      - OPT(i-1)(j)
      - OPT(i)(j-1)
    - Thus: `max( OPT(i-1)(j), OPT(i)(j-1) )`

- Case 2: If first_string[i] == second_string[j]
    - This case is basically saying if the characters are the same.
    - Using the example, if `i = 6` and `j = 3`. Where the highlighted 
      character is denoted using `| |`
      `AABCXY   |Z|`
      `AXY |Z| ABC`
    - Same as above, similar to 1 dimension, but now we have 3 sub-cases
        - OPT(i-1)(j-1) + 1
            - This is saying the optimal solution of `AABCXY` and `AXY`
              which is `AXY`, which is 3, plus 1 (which is `Z`).

-  Base Case: Literally the start, However, we need two sets of base cases
   because we are comparing two strings
    - if i = 0, then we are comparing all j with an empty string.
      - Specifically, OPT(0)(j) is saying the optimal solution for an string
        length 0 and a string of length j.
    - same for if j = 0
    - Therefore:
        - when i = 0, OPT(0)(j) = 0 for all j
        - when j = 0, OPT(i)(0) = 0 for all i

### Putting it together

```txt
OPT(i)(j) = { 0                              if i = 0 or j = 0
          = { max(OPT(i-1)(j), OPT(i)(j-1))  if string_one[i] != string_two[j]
          = { OPT(i-1)(j-1) + 1              if string_one[i] == string_two[j]
```

# Application in Biology (RNA Stuff)



# Application in Graphs (Shortest Path)

Given a Graph G(V,E), we want to find the shortest path from node s to node t.

To do this we assume we have a solution:

OPT(i, v) = the shortest path from node v to node t in less than or equal to i 
            edges.

- Case 1: There exists a solution with less edges
    - OPT(i-1, v)
    - This is saying there is a path from node v to node t that exists in 
      i-1 edges.

- Case 2: There exists a node w such that path w to t uses i - 1 edges and
          v is connected to w in 1 edge.
    - OPT(i-1,w) + cost_vw
    - More specifically its `min(OPT(i-1,w) + cost_vw)`
    - Think about it like this:
        - There exists multiple paths from w to t, if we be greedy and just
          take the minimum of this path, we aren't looking at the overall
          solution. Hence, we want the minimum OVERALL cost of path w to t
          plus teh cost from v to w.

- Base Case: 
    - OPT(0,t) = 0
        - This is aying that if we are at t, then cost to reach it is 0
    - OPT(0,v) = inf
        - This is saying, if we are at node v but have 0 edges to reach t
          then its impossible to reach the goal node, hence its inf. Assuming
          that v != t.

Putting it together, we get:
```txt
OPT(i,v) = { 0                                            if i = 0 and v == t 
           { inf                                          if i = 0 and v != t
           { min(OPT(i-1,v), min(OPT(i-1,w) + cost_vw))   otherwise
```

