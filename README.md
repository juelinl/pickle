# Pickle
Research on efficient approximate nearest neighbor search (ANNS).

# Direction I
Ideally, an efficient index would have both a small memory footprint and an optimized memory layout for ANNS. (A good example would be B-tree for SQL database.)
However, graph-based indices (etc. HNSW and NND) have a small memory footprint but they suffer from small random memory accesses, whereas clustered indices (SPANN and SPFresh) suffer from excessive memory footprint when compared to graph-based indices.

Thus, we ask the question: Can graph index and cluster index have a happy marriage? 

## TLDR
The main challenge is to build an index with an optimized memory layout without increasing memory footprints significantly. 
It turns out to be a very difficult challenge.

## Observation & Motivation

1. Distance computation is one of the main bottlenecks of ANNS.
2. Distance computation efficiency is sensitive to memory layout.
3. Graph index relies on small random memory accesses.
4. Cluster index suffers from excessive memory load.

## Approach: Graph Index With Hub Nodes
A potential solution is to use a graph-based index with hub nodes. In this solution, the graph index is extended with hub nodes, which have two key properties: they have high degree and are frequently visited by many queries. The neighborhood of a hub node is stored as a bucket containing the vectors of all the neighboring nodes. This enables efficient batched distance computation implemented in terms of matrix multiplication. These vectors may be replicas of the primary copy of the node vector, which may be stored in another memory location outside of the bucket.

### Challenge: How to select nodes to be part of the hub nodes?

#### Hypotheses 
1. Nodes in the top layers of the HNSW index might be good candidates.
2. Graph access might be skewed, so frequently access nodes can be hub nodes.
3. The K-means centroids of the graph might be good candidates.

#### Experiment

#### Analysis 

#### Hunch


# Research Direction II
Efficient index construction is becoming a real challenge, especially for web-scale datasets. A single host disk-based algorithm (DiskANN) can take days to construct the index for a dataset with 1B nodes. This is where the distributed system comes to the rescue.
Many interesting questions arise here:
1. Shall we partition the graph (share nothing) or use shared memory to scale the graph search?
2. Shall we construct one big index that has many edges or several randomized small indexes and merge their search results? 

## TLDR

# Observation & Motivation
1. The index construction for graph-based indexes is slow.
2. The index construction memory overhead can be high.
3. IVF is fast during construction but slow during the search.

# Research Direction III
What if the distances between the query and the documents are computed using Transformers (BERT)?
This makes things interesting because the embedding of the documents cannot be easily computed offline without knowing the query.
ColBERT tackles it by computing the embedding at the token rather than document level. Can we do better than this?