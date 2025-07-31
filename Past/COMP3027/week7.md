# Network Flow (Application)

## Matching

This is basically creating a set of edges such that no node appears twice.

So given this graph:
```txt
A --- B
C --- D
E --- F
```

Let M be a set of matching edges:

`M = {(A,B), (C,D)}`

As you can see here, no node appears twice.

Another valid M could also be:

`M = {(A,B), (C,D), (E,F)}`

We can anyhow add edges, however, for it to be matching, there must not be
a node that appears twice.

Matching is bad term for this because it implies pairs. However, adding
edges is NOT pairing them up. 

## Biparte Matching

This is just matching but in a biparte graph

However, to do actually APPLY this and create an algorithm for this is 
a little more complex.

We can reduce this Biparte Matching problem to a Flow problem.

### Step 1 Tranforming into a Flow Graph

In having a biparte graph, we can add a node on the far LHS being the
source node (connected to all nodes on the LHS). Similarly, we can add a
node to the RHS being the sink node (connected to all nodes on the RHS).

As a Biparte Graph is normally undirected, we just transform all edges into
a Left-to-Right direction and give each edge a wieght of 1. 

### Step 2 Flow Algorithm

We can use a max flow algorithm like the Edmond's Karp Algorithm

### Step 3 Extract Matching from Flow

Using the solutions (all valid paths) from step 2, we can get all matching
edges in the original graph.

### Running Time

Step 1 runs in O(n + m) time

Step 2 runs in O(n * m^2) time

Step 3 runs in O(n) time

## Perfect Matching

> [!note] NOTE
> Edges are generally denoted as (a,b) where a and b are nodes.

For Matching to be considered "Perfect", all nodes must be accounted for.
In other words, not a single node must be left out.

An example would be, in a biparte graph, if `max_flow = count(LHS_nodes)` then
it is considered "Perfect Matching".

This is because we set all edge weights (capacities) to 1.

### Marriage Theorem

Every node on the Left-Hand Side (LHS) must be matched to a unique node on 
the Right-Hand Side (RHS).

This does not apply to all biparte graphs, but must be true for Perfect 
Matching to be true.

## Edge Disjoint Paths

Edge disjoint paths are paths that do not share any common edges.

They can share nodes but NOT edges.

## Network Connectivity

This is basically just min cut, but instead of flow (edge weight/capacity)
its te minimum number of edges. 

In other words, if we set all edge weights to 1, we want the minimum number
of cuts to separate s from t.

## Combining EDP and NC

By logic now, the number of edge disjoint paths (unique paths from s to t,
such that no edge is repeated) is also the minimum number of cuts needed
to separate s from t.




