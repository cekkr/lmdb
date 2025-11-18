# To implement:
- Add as high priority reference the cluster-ready algorithms structure (the tree-fork data approach could work in a distributed scenario)
- Implement a "matrix-related" tree-series predictions (Go to "Matrix related tree")

## Matrix-related tree
Cheetah uses trees structure to divide bytes in well-defined boundaries fast access files. Here's leans the possibility to extend its basic usage for prediction purposes.
But predictions are not dynamic if there is no way to relate probabilities to a (dynamic) context matrix.
Here is needed to define a particular type of table: the context_prediction. 
A context_prediction tables:
- Can returns multiples results for the same series of bytes
- Can evaluate multiple windows of a byte array in base of query options
- Is able to change probabilities in base of current reference context, representend by a "matrix" (that sub-arrays can have different sizes)
- Discard low-probability connections: especially during training, first connections are "lousy", and storing all connections would occupy gigabytes of storage. A real time command/automatic limit for discarding irrilevant connection is needed. 

Todo: describe the definition of context-matrix, then explain how to make it working in a computational efficient way. 'Cause now I'm too tired.