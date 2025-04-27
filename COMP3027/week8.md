# Circulation and Nondeterministic Polynomial time (NP) and Intractibility

## Circulation

This is simply an extension of Flow.

The difference is, instead of being given a source and sink node, we are 
given multiple mini-source and mini-sink nodes.

Note that each mini-source can only give a maximum of XX flow and mini-sink 
nodes can only consume XX flow. 

Other nodes would have a "conditional" status where they MUST have at least 
XX flow passing through them.

A valid circulation is found when enough flow can meet the total sink 
requirements.

Basically say I have 2 nodes X and Y that can output 10 flow total. And I have 
2 sink nodes A and B such that they can absorb 5 flow total.

As a result only 1 of 2 things can happen:

> Assume that the flow traverses the graph as optimally as possible 

1. A flow that is >= 5 can reach the sink nodes
    - This means that it is a valid circulation as the demand is reached

2. A flow that is < 5 can reach the sink nodes
    - This means that it is NOT a valid circulation as the demand is NOT reached.

This can be reduced to a max flow graph simply by adding a "super source" and 
a "super sink" node. The super sink node would connect to all "mini sink" 
nodes and the super source node would connect to all "mini source" nodes and 
their edge weights is equal to the amount the supply/demand. 

After this, we run the max flow algorithm to determine max flow, and if:

> Assuming all condtions are met (nodes that require XXX flow)

- max_flow < total_demand: graph is not a valid circulation
- max_flow >= total_demand: graph is a valid circulation

## Nondeterministic Polynomial time (NP)

Commonly abbbreviated as NP, (Nondeterministic Polymonial), refers to an 
algorithm that allows us to determine if a given solution is valid in 
polynomial time (n^c time).

In simpler terms, you can check a solution quickly (polynomial time), but 
you might not know how to solve it.

## Descision vs Search vs Optimization

Most problems can be split into 3 main types:
1. Descision Problem: Is this a valid solution? (returns yes or no)
2. Search Problem: Here is a solution (returns a solution)
3. Optimization Problem: This is the best solution (returns the best solution)

> [!note]
> In theory, we can reduce between these types in polynomial time:
> Decision ≤ Search ≤ Optimization ≤ Decision

### Theoretical Reduction

- If you can decide if a good solution exists (Decision), you can search by 
  trying different choices.
- If you can search for a good enough solution, you can optimize by searching 
  cleverly.
- If you can optimize to get the best, you can decide by checking if the best 
  is above a certain threshold.

Hence, we get this cycle:
- Decision can be reduced to Search in polynomial time.
- Search can be reduced to Optimization in polynomial time.
- Optimization can be reduced to Decision in polynomial time.

## Intractibility

Intractable problems = Problems we believe cannot be solved fast 
(polynomial time, like n^2 or n^5).

## Self Reducibility and Downward Self Reducibility

Examples:
Say I have a search problem. I can reduce this to a descision problem by 
asking multiple descision questions until I build a solution!






