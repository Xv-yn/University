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
solution.

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





