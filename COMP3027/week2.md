# Reduction
Say I have Problem A

Problem A is basically the same as Problem B

Problem B can be solved using Algorithm B

Now I have the Solution for Problem B

Since Problem B is basically the same as Problem A

I now have the solution for Problem A

# Greedy Algorithm
The essence of greedy algorithm is to take the best option first, regardless
if there is a better overall solution.

At this point, it would just be sorting algorithm (take from lowest first)

## E.g. Interval Scheduling

Order by:
- Start Time
- End Time
- Process Duration
- Fewest Conflicts

## Analysis
1. Define the greedy solution
2. Compare solutions if Greedy_solution != optimal_solution then find
   the first solution
3. Exchange pieces, by transforming optimal_solution that is "closer" to
   the greedy_solution and prove that cost doesn't increase
4. Iterate until optimal_solution = greedy_solution

## E.g. Knapsack Problem

Order by:
- Most benefit
- Least weight
- highest benefit to weight ratio

## Scheduling to Minimize Lateness

Because due times can be unique, it forces a certain order.
Goal: Schedul all jobs to minimize maximum lateness

Order by:
- Shortest Processing Time
- Smallest Slack 
    - slack = due_time - processing_time
- Earliest Deadline First
    - If greedy strategy, it ignores processing time


