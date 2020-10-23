from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import MultiVerify


def test_graph_sage_mean(default_plugin_resolver):
    """
== Training Subgraph ==

[Training Subgraph A (fully connected), nodes 0..9] --------------|
                           |                                      |
                    node 9999_09_10                               |
                           |                                      |
[Training Subgraph B (fully connected), nodes 10..19]     node 9999_29_00
                           |                                      |
                    node 9999_19_20                               |
                           |                                      |
[Training Subgraph C (fully connected), nodes 10..19] -------------

Training Subgraph A nodes all have feature vector [1, 0, 0, 0, 0, ..., 0]
Training Subgraph B nodes all have feature vector [0, 1, 0, 0, 0, ..., 0]
Training Subgraph C nodes all have feature vector [0, 0, 1, 0, 0, ..., 0]
Nodes 9999_09_10, 9999_19_20, and node 9999_29_00 have the zero vector as their node features.



== Testing Subgraph ==

[Testing Subgraph A (fully connected), nodes 8888_00..8888_19]
                        |
                 node 8888_00_20
                        |
[Testing Subgraph B (fully connected), nodes 8888_20..8888_49]

Testing Subgraph A nodes all have feature vector [1, 0, 0, 0, 0, ..., 0]
Testing Subgraph B nodes all have feature vector [0, 1, 0, 0, 0, ..., 0]
Node 8888_00_20 hsa the zero vector as a its node features.



== Differences Between Training & Testing Graphs ==

All the complete subgraphs in the training graph have 10 nodes, but the complete subgraphs in the testing graph do NOT.

The test verifies for the testing graph that the 20 nearest neighbors in the embedding space of each node are all part of the same complete subgraph.
    """
    dpr = default_plugin_resolver

    # Generate Training Graph
    a_nodes = np.arange(10)
    b_nodes = np.arange(10, 20)
    c_nodes = np.arange(20, 30)

    complete_graph_a = nx.complete_graph(a_nodes)
    complete_graph_b = nx.complete_graph(b_nodes)
    complete_graph_c = nx.complete_graph(c_nodes)

    nx_graph = nx.compose(
        nx.compose(complete_graph_a, complete_graph_b), complete_graph_c
    )
    nx_graph.add_edge(9999_09_10, 9)
    nx_graph.add_edge(9999_09_10, 10)
    nx_graph.add_edge(9999_19_20, 19)
    nx_graph.add_edge(9999_19_20, 20)
    nx_graph.add_edge(9999_29_00, 29)
    nx_graph.add_edge(9999_29_00, 0)

    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    mv = MultiVerify(dpr)

    embedding_size = 50
    node_feature_nodes = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.arange(33),
        node_ids=np.concatenate(
            [np.arange(30), np.array([9999_09_10, 9999_19_20, 9999_29_00])]
        ),
    )
    node_feature_np_matrix = np.zeros([33, embedding_size])
    node_feature_np_matrix[a_nodes, 0] = 1
    node_feature_np_matrix[b_nodes, 1] = 1
    node_feature_np_matrix[c_nodes, 2] = 1
    node_feature_np_matrix[30:] = np.ones(embedding_size)
    node_feature_matrix = dpr.wrappers.Matrix.NumpyMatrix(node_feature_np_matrix)
    node_features = dpr.wrappers.NodeEmbedding.NumpyNodeEmbedding(
        node_feature_matrix, node_feature_nodes
    )

    # Run GraphSAGE
    walk_length = 5
    walks_per_node = 1
    layer_sizes = dpr.wrappers.Vector.NumpyVector(np.array([40, 30]))
    samples_per_layer = dpr.wrappers.Vector.NumpyVector(np.array([10, 5]))
    epochs = 35
    learning_rate = 5e-3
    batch_size = 2

    assert len(layer_sizes) == len(samples_per_layer)

    embedding = mv.compute(
        "embedding.train.graph_sage.mean",
        graph,
        node_features,
        walk_length,
        walks_per_node,
        layer_sizes,
        samples_per_layer,
        epochs,
        learning_rate,
        batch_size,
    ).normalize(dpr.types.GraphSageNodeEmbedding.StellarGraphGraphSageNodeEmbeddingType)

    # Create Testing Graph
    unseen_a_nodes = np.arange(8888_00, 8888_20)
    unseen_b_nodes = np.arange(8888_20, 8888_50)
    unseen_complete_graph_a = nx.complete_graph(unseen_a_nodes)
    unseen_complete_graph_b = nx.complete_graph(unseen_b_nodes)
    unseen_nx_graph = nx.compose(unseen_complete_graph_a, unseen_complete_graph_b)
    unseen_nx_graph.add_edge(8888_00_20, 8888_00)
    unseen_nx_graph.add_edge(8888_00_20, 8888_20)
    unseen_node_feature_np_matrix = np.zeros([51, embedding_size])
    unseen_node_feature_np_matrix[0:20, 0] = 1
    unseen_node_feature_np_matrix[20:50, 1] = 1
    unseen_node_feature_matrix = dpr.wrappers.Matrix.NumpyMatrix(
        unseen_node_feature_np_matrix
    )
    unseen_node_feature_nodes = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.arange(51),
        node_ids=np.concatenate(
            [unseen_a_nodes, unseen_b_nodes, np.array([8888_00_20])]
        ),
    )
    unseen_node_embedding = dpr.wrappers.NodeEmbedding.NumpyNodeEmbedding(
        unseen_node_feature_matrix, unseen_node_feature_nodes
    )
    unseen_graph = dpr.wrappers.Graph.NetworkXGraph(unseen_nx_graph)
    matrix = mv.transform(
        dpr.plugins.metagraph_stellargraph_stellargraph.algos.util.graph_sage_node_embedding.apply,
        embedding,
        unseen_graph,
        unseen_node_embedding,
        batch_size=batch_size,
        worker_count=1,
    )

    # Verify GraphSAGE results
    def cmp_func(matrix):
        assert tuple(matrix.shape) == (51, layer_sizes.as_dense(copy=False)[-1])
        np_matrix = matrix.as_dense(copy=False)
        unseen_a_vectors = np_matrix[0:20]
        unseen_b_vectors = np_matrix[20:50]

        _, neighbor_indices = (
            NearestNeighbors(n_neighbors=20).fit(np_matrix).kneighbors(np_matrix)
        )

        for unseen_a_node_position in range(20):
            unseen_a_node_neighbor_indices = neighbor_indices[unseen_a_node_position]
            for unseen_a_node_neighbor_index in unseen_a_node_neighbor_indices:
                assert 0 <= unseen_a_node_neighbor_index < 20

        for unseen_b_node_position in range(20, 50):
            unseen_b_node_neighbor_indices = neighbor_indices[unseen_b_node_position]
            for unseen_b_node_neighbor_index in unseen_b_node_neighbor_indices:
                assert 20 <= unseen_b_node_neighbor_index < 50

    matrix.normalize(dpr.types.Matrix.NumpyMatrixType).custom_compare(cmp_func)
