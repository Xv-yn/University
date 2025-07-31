# Network Flows (Theory)

The essence of Network Flow is basically imagining a given graph as a
water pipe network. Each edge is a pipe that can send a maximum amount
of water. Each node is a tank, that lets us redirect water to connected pipes.

Given some directed graph. Lets say we want to send as much water as possible
from node s to node t.

Now say we have the following network (directed graph):
```txt
        (2)--9->(5)
       -^| \     | \
      /  |  \    |  \
    10   4  15  15  10
    /    |    \  |    \
   /     v     `vv     `v
(S)--5--(3)--8->(6)-10->(T)
   \     |^-     |     -^
    \    |  \    |    /
    15   4   6  15  10
      \  |    \  |  /
       `vv     \ v /
        (4)-----(7)
            30
```

In this directed graph, each edge is the maximum capacity.

The max flow is the maximum amount of water possible to send from
source to sink.

We can do this by sending a maximum flow through one of the source pipes 
until we hit a bottle neck. And keep going with different pipes and different
paths. This is solving it intuitively.

The minimum cut is the minimum capacity edges we need to cut to separate
s from t. 

For example, using the above, if we cut all pipes from S,being 10, 5 and 15
we get a minimum cut of 30. But we can do better.

## Logical Method

This is a logical method (informal algorithm) to solve this problem.

We first pick a path that we can use to send flow from source to sink. 

At some point, we may or may not hit a bottleneck.

We generate what is called a Residual Network. This is basically a backtrack
of paths + leftover capacity map. In the original map, if there is still 
space to add more flow, we send the original flow backwards and the empty 
space forwards. 

Using the residual network, we find any more valid paths from s to t and add
it to the max flow.

There exists a theorem called 'Weak Duality' where the maximum flow must be
less than or equal to the minimum cut.

The minimum cut is basically we split the graph into a biparte graph
such that the LHS only contains the source side nodes and the RHS contains
the sink side nodes. We only add the source-to-sink edges and NOT the
sink-to-source edges. 

After finding the max flow, we attempt to find a path from source to sink.
There will not be a valid path, but all nodes reachable from the source 
can be grouped into the LHS of the biparte graph and the rest into the RHS.

## Augmenting Path Algorithm

This is the most simpe an intuitive algorithm. This is basically keep pushing
flow through each edge until you hit a bottleneck. Any excess flow goes 
through another valid edge until you hit the sink. 

## Ford-Fulkerson Algorithm

This algorithm is quite literally the logical method. But what it does is
initialize all flow to 0 initially. Then create a residual network.

Then it finds a path using the residual network and updates the max flow
and residual netowrk at the same time. And repeat until no more paths
can be found via the residual network. And done!

This runs in O(max_flow * E) time

## Edmond's Karp

This is a more efficient method than the Ford-Fulkerson Algorithm where
it uses BFS to find the path from s to t in the least number of edges
when doing the augmenting paths.

As a result, this runs in O(V * E^2) time.

# Terminology
Satistied = when the maximum amount of flow meets the capacity

Bottleneck = the edge which limits the flow

Source = (start node) where the water comes out

Sink = (destination node) where the water should go

Conservation = If water goes in, it must come out (unless source or sink)


