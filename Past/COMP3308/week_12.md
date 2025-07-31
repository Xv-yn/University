# Clustering

Clustering – the process of partitioning the data into a set of groups
(called clusters) so that the items from the same cluster are:
- similar to each other
- dissimilar to the items in other clusters

> [!note]
> Similarity is defined in terms of distance measure


Clustering is unsupervised learning: no labels

Given:
- A set of unlabeled examples (input vectors) x_i
- k – desired number of clusters (may not be given)
- Task: Cluster (group) the examples into k clusters (groups)

> [!caution] IMPORTANT
> - Supervised: We know the class labels and the number of classes. We
> want to build a classifier that can be used to predict the class of new
> (unlabelled) examples.
> - Unsupervised: We do not know the class labels and may not know the
> number of classes. We want to group similar examples together.

A good clustering will produce clusters with
- High cohesion ( i.e. high similarity within the cluster)
- High separation (i.e. low similarity between the clusters)

Cohesion and separation are measured with a distance function

## K-Means Clustering Algorithm

1. Choose k examples as the initial centroids (seeds) of the clusters
2. Form k clusters by assigning each example to the closest centroid
3. At the end of each epoch:
  - Re-compute the centroid of the clusters
  - Check if the stopping criterion is satisfied: centroids do not change. If
    yes – stop; otherwise, repeat steps 2 and 3 using the new centroids.


# Terminology

Centroid (means) – the “middle” of the cluster
- Does not need to be an actual data point in the cluster

Medoid M – the centrally located data point in the cluster
- Must be a datapoint int he cluster
- The datapoint closest to the Centroid

Single link (MIN) – The smallest pairwise distance between elements
from each cluster
- Given two clusters, find two points (one from each) such that the distance
  is the SMALLEST

Complete link (MAX) – the largest pairwise distance between elements
from each cluster
- Given two clusters, find two points (one from each) such that the distance
  is the LARGEST

Average link – the average pairwise distance between elements from each
cluster

