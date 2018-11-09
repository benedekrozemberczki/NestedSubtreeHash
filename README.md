# Nested Subtree Hashes

<p align="justify">
Nested subtree hashing is an embedding algorithm which learns representations for a set of graphs by hashing the Weisfeiler-Lehman subtree patterns. The procedure places graphs in an abstract feature space where graphs with similar structural properties (Weisfehler-Lehman features) are clustered together. Nested subtree hashing has a linear runtime complexity in the number of graphs in the dataset which makes it extremely scalable. At the instance level creating a graph representation has a linear runtime and space complexity in the number of edges. This specific implementation supports multi-core data processing in the feature extraction and hashing phases. (So far this is the only implementation which support multi-core processing in every phase).
</p>
<p align="center">
  <img width="720" src="graph_embedding.jpeg">
</p>

This repository provides an implementation for Nested Subtree Hashing as it is described in:
> Nested Subtree Hash Kernels for Large-scale Graph Classification Over Streams
> Bin Li, Xingquan Zhu, Lianhua Chi, Chengqi Zhang
> IEEE 12th International Conference on Data Mining.

### Requirements

The codebase is implemented in Python 2.7. Package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
pandas            0.23.4
jsonschema        2.6.0
joblib            0.13.0
numpy             1.14.3
```

### Datasets

The code takes an input folder with json files. Every file is a graph and files have a numeric index as a name. The json files have two keys. The first key called "edges" corresponds to the edge list of the graph. The second key "features" corresponds to the node features. If the second key is not present the WL machine defaults to use the node degree as a feature.  A sample graph dataset from NCI1 is included in the `dataset/` directory.

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --input-path      STR     Input folder.         Default is `dataset/`.
  --output-path     STR     Embeddings path.      Default is `features/nci1.csv`.
```
#### Model options
```
  --dimensions      INT     Number of dimensions.                          Default is 16.
  --workers         INT     Number of workers.                             Default is 4.
  --wl-iterations   INT     Number of feature extraction recursions.       Default is 2.
```

### Examples

The following commands learn an embedding of the graphs and writes it to disk. The node representations are ordered by the ID.

Creating a Nested Subtree Hash embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.

```
python src/main.py
```

Creating an embedding of an other dataset. Saving the output in a custom place.

```
python src/main.py --input-path new_data/ --output-path features/nci2.csv
```

Creating an embedding of the default dataset with 3x32 dimensions as each recursion creates a 32 dimensional multi-scale subspace.

```
python src/main.py --dimensions 32 --wl-iterations 3
```
