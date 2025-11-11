This the base code of project "cheetah" for a ultra-rapid fast access database. It has to be re-adapted to work as database engine for LMDB project with the name of "cheetah-mldb". 

At its base, cheetah allows to extremly fast access to data from a key using a perfectly compartmentalized key cataloging by byte size, fast tree access algorithms, and various levels of caching.

For the "lmdb" version to be implemented, it must be adapted as best as possible to the needs of the Python project, further implementing:
- Rapid statistical computation (and associated caching)
- Rapid key contextualization: this requires byte-based tokenization of values ​​and efficient auto-sorting of dynamic sets (more specific probabilities can be obtained by delving deeper into the context of the prediction, without requiring ad hoc retraining on the Python side) of vector matrices and associated probabilistic links.
- This requires brute-force analysis of the data relationships in the DB (as well as efficient single-ordering of matrices to maintain fixed references), maximizing the rapid data access speed and the use of multiple hard disk files to retrieve and compare data and obtain more relevant statistical results.