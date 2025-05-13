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

Commonly abbbreviated as NP, (Nondeterministic Polynomial), refers to an 
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

Reduction from problem X to problem Y is denoted as follows:

X ≤ Y

This feels counter intuitive, but it isn't when you give it deeper thought.

Now we say that problem X POLYNOMIAL reduces to problem Y when:
- INSTANCES of problem X:
    - Can be solved in polynomial time
    - "translating" back and forth from problem X to problem Y runs in 
      polynomial time.

Now thinking about this logically, if I can solve problem Y in polynomial 
time, then I would also be able to solve problem X in polynomial time!

Another piece that seems obvious, if I CANNOT solve problem X in polynomial 
time, then I also cannot solve problem Y in polynomial time.

Examples:
Say I have a search problem. I can reduce this to a descision problem by 
asking multiple descision questions until I build a solution!

> [!note]
> Descision algorithms can be further reduced to another descision algorithm:
>   - Given this "solution", is this solution valid?

## Triple Known Problems

### Vertex Cover

- Given a graph get the smallest list of nodes such that all edges are 
  accounted for

### Independent Set

- Given a graph get the smallest list of nodes where no nodes are neighbours

> [!note]
> Vertex Cover and Independent Set are complementary

### Set Cover

- Given a bunch of sets, find the minimum number of sets such that their union 
  is equal to all nodes in a given graph.

We can reduce the Vertex Cover problem into the Set Cover problem

Say we have some graph, and we want to apply the vertex cover. We can reduce 
this problem to the Set Cover, where we generate lots of combinations of 
sets containing nodes from the graph. We find the minimum number of unions 
needed to get all nodes in the graph. This minimum number of unions is equal 
to the number of nodes needed for the Vertex Cover problem.

## Polynomial

### P

P stands for polynomial

More specifically, P is basically a group/class of algorithms that meet the 
following criteria:
- Solvable in Polynomial time

Example:

"Is Waldo at location(x,y)?"

We can solve this by iterating through each cell in location(x,y) in 
polynomial time to determine YES/NO if Waldo is at location(x,y).

### NP

NP stands for Nondeterministic Polynomial

Now this is a bit more confusing. NP is a group/class of algorithms that meet 
the following criteria:
- Is a decision problem
- Its `certifier` can be solved in polynomial time
    - "Is Waldo at location(x,y)?" is a certifier
    - A `certifier` takes in a potential solution and verifies the solution
- Takes in a "normal" input e.g. a graph, a picture, etc. 

Example:

"Is Waldo in this picture?"

To prove that this problem is in NP, we take a YES/NO instance of this problem 
and create a certifier: "Is Waldo at location(x,y)?". Because of previous 
reasoning, we know that the certifier is in P. Because the certifier can be verified in 
polynomial time, "Is Waldo in this picture?" is in NP.

### NP-Hard

NP-Hard stands for Nondeterministic Polynomial Hard

These problems are NOT decision problems and are a group/class of algorithms that meet 
the following criteria:
- Not required to be a decision problem
- Not required to be verifiable in polynomial time

Example:

"Where is Waldo in this picture?"

This problem asks for a location, making it a search problem. To prove that this problem 
is in NP-Hard, we try to reduce a known NP-Hard problem (3SAT, etc.) into this problem.

Once we have a sucessful reduction, we have proven that this problem is in NP-Hard.

### NP-Complete

NP-Complete stands for Nondeterministic Polynomial Complete

These problems are a group/class of algorithms that meet the following criteria:
- Problem is in NP
- Problem is in NP-Hard

Example:

"Is Waldo in this picture?"

- We have proven that this problem is in NP
- IF we can prove that this problem is in NP-Hard by reducing a known NP-Hard problem 
  into this problem. 
  - This problem is NP-Complete

