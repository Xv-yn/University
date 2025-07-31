# Dynamic Programming

## Key Steps
1. Break the problem into smaller overlapping subproblems
2. Establish a relationship between smaller and larger subproblems
3. Solve Base Cases

> [!note] NOTE
> This is SIMILAR to Divide and Conquer but IT IS DIFFERENT

## Example

Given the array A of n numbers, we want the maximum sum found in any
contiguous subarray (a zero-length subarray has a maximum of 0)

A = \[1,-2,7,5,6,-5,5,8,1,-6\]

1. Defining Subinstances (Breaking the Problem)

OPT(i) = value of optimal solution ending at i (Can be Empty)

```txt
          SUM   1  -2  7  5  6  -5  5  8  1  -6
 OPT[1] = 1     1
 OPT[2] = 0     1  -2
 OPT[3] = 7     1  -2  7
 OPT[4] = 12    1  -2  7  5
 OPT[5] = 18    1  -2  7  5  6
 OPT[6] = 13    1  -2  7  5  6  -5
 OPT[7] = 18    1  -2  7  5  6  -5  5
 OPT[8] = 26    1  -2  7  5  6  -5  5  8
 OPT[9] = 27    1  -2  7  5  6  -5  5  8  1
OPT[10] = 21    1  -2  7  5  6  -5  5  8  1  -6
```

2. Find Recurrences

There are 2 main cases:
* `A[i]` is not included in the optimal solution ending at i
    * The best solution is at index i - 1. 
      We can add the contents of index i, but it won't make a difference
* `A[i]` is included in the optimal solution
    * We have a good solution at index i - 1, but adding the contents of
      index i will make it a better solution

3. Solving the Base Case

At the very base case, `max(A[1],0)` is the the most optimal solution
if i = 1. We use the `max()` function just in case `A[1]` is negative.

Hence, by relying on the 2nd main case in step 2, we can just iterate through
the array once.

```txt
OPT[1] = max(A[1],0)    # This is solving the base case

for i = 2 to n do
    OPT[i] = max(0,OPT[i-1] + A[i])    # This iterates through the rest of the list

MaxSum = OPT
```
## Comparing Against Divide and Conquer

When doing divide and conquer, while time complexities can range, 
we generally use recursion break the problem, leaving us with log n time. 
However, We do need to recursively merge, leading to an n log n time compelxity

In the case of Dynamic Programming, the time compelxities can also range. 
However, in cases like above, it can allow for a much faster time complexity
as we only really need to iterate through the list once, leaving us with n
time complexity.

## One Advantage over Divide and Conquer

Greedy Algorithms may work for weighted interval scheduling problems.
However, there is no clear answer because some do and some don't.

However, dynamic programming algorithms definitely work!

Using an example of weighted interval scheduling:

> [!important] IMPORTANT SAMPLE ANSWER

Assumptions:
- Jobs are sorted by finish times

Facts:
1. Optimal Solution either includes last job or not
    - If so, then the optimal solution for the remaining jobs are compatible
      with the last job
    - Else, it is the optimal solution for remaining jobs
2. Subinstances are prefixes (meaning that we build up validity of the solution
   one step at a time)
    - E.g. since job 1 is valid, if we add job 2 and it is valid, then OPT 2
      is valid.

Step 1: Defining Subinstances
- let `OPT[j]` be the optimal solution consisting of jobs from 1 to j

Step 2: Find Recurrences
We do either of the two things (and keep the best):
* Case 1: We include job j
    - We add it's weight w_j
    - Value = w_j + OPT(p(j))
* Case 2: We don;t include job j
    - Which is just the previous solution
    - Value = OPT(j-1)
Thus, the full recurrence (because we want the maximum sum of subset) is:

OPT(j) = max( w_j + OPT(p(j)), OPT(j-1))


