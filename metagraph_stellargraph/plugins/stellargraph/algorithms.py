from .. import has_stellargraph
from metagraph import concrete_algorithm
from typing import Tuple

if has_stellargraph:
    import tempfile
    import os
    import uuid
    import stellargraph as sg
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from metagraph.plugins.networkx.types import NetworkXGraph
    from metagraph.plugins.python.types import PythonNodeMap, PythonNodeSet
    from metagraph.plugins.numpy.types import (
        NumpyNodeMap,
        NumpyVector,
        NumpyMatrix,
    )
    from .types import StellarGraph, StellarGraphGraphSageNodeEmbedding

    @concrete_algorithm("subgraph.extract_subgraph")
    def sg_extract_subgraph(graph: StellarGraph, nodes: PythonNodeSet) -> StellarGraph:
        # TODO StellarGraph.subgraph can take any iterable, not necessarily a python set
        subgraph = graph.value.subgraph(nodes.value)
        return StellarGraph(
            subgraph,
            node_weight_index=graph.node_weight_index,
            is_weighted=graph.is_weighted,
            node_sg_type=graph.node_sg_type,
            edge_sg_type=graph.edge_sg_type,
        )

    @concrete_algorithm("clustering.connected_components")
    def sg_connected_components(graph: StellarGraph) -> PythonNodeMap:
        index_to_label = dict()
        for i, nodes in enumerate(graph.value.connected_components()):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodeMap(index_to_label)

    @concrete_algorithm("util.graph_sage_node_embedding.apply")
    def sg_graph_sage_node_embedding_apply(
        embedding: StellarGraphGraphSageNodeEmbedding,
        graph: NetworkXGraph,
        node_features: NumpyMatrix,
        node2row: NumpyNodeMap,
        batch_size: int = 1,
        worker_count: int = 1,
    ) -> NumpyMatrix:
        # TODO generating a whole new StellarGraph here seems expensive
        node_features_df = pd.DataFrame(
            node_features.as_dense(copy=False), index=node2row.pos2id
        )
        sg_graph = sg.StellarGraph.from_networkx(
            graph.value,
            edge_weight_attr=graph.edge_weight_label or "weight",
            node_features=node_features_df,
        )
        node_gen = sg.mapper.GraphSAGENodeGenerator(
            sg_graph, batch_size, embedding.samples_per_layer
        ).flow(nodes)

        node_embeddings = embedding.model.predict(
            node_gen, workers=worker_count, verbose=1
        )
        return NumpyMatrix(node_embeddings)

    @concrete_algorithm("embedding.train.node2vec")
    def sg_node2vec_train(
        graph: StellarGraph,
        p: float,
        q: float,
        walks_per_node: int,
        walk_length: int,
        embedding_size: int,
        epochs: int,
        learning_rate: float,
        worker_count: int = 1,
        batch_size: int = 10_000,
    ) -> Tuple[NumpyMatrix, NumpyNodeMap]:

        walker = sg.data.BiasedRandomWalk(
            graph.value, n=walks_per_node, length=walk_length, p=p, q=q,
        )
        unsupervised_samples = sg.data.UnsupervisedSampler(
            graph.value, nodes=list(graph.value.nodes()), walker=walker
        )
        generator = sg.mapper.Node2VecLinkGenerator(graph.value, batch_size)

        node2vec = sg.layer.Node2Vec(embedding_size, generator=generator)
        x_inp, x_out = node2vec.in_out_tensors()
        prediction = sg.layer.link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
        )(x_out)
        model = tf.keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.binary_crossentropy,
            # metrics=[tf.keras.metrics.binary_accuracy],
        )
        model.fit(
            generator.flow(unsupervised_samples),
            epochs=epochs,
            use_multiprocessing=False,
            workers=worker_count,
            shuffle=True,
        )

        nodes = graph.value.nodes().sort_values()
        node_gen = sg.mapper.Node2VecNodeGenerator(graph.value, batch_size).flow(nodes)
        x_inp_src = x_inp[0]
        x_out_src = x_out[0]

        embedding_model = tf.keras.Model(inputs=x_inp_src, outputs=x_out_src)
        node_embeddings = embedding_model.predict(node_gen, workers=worker_count)

        node2index = NumpyNodeMap(np.arange(len(nodes)), node_ids=nodes.to_numpy())
        matrix = NumpyMatrix(node_embeddings)
        return (matrix, node2index)

    @concrete_algorithm("embedding.train.graphwave")
    def sg_graphwave_train(
        graph: StellarGraph,
        scales: NumpyVector,
        sample_point_count: int,
        sample_point_max: float,
        chebyshev_degree: int,
    ) -> Tuple[NumpyMatrix, NumpyNodeMap]:
        sample_points = np.linspace(0, sample_point_max, sample_point_count).astype(
            np.float32
        )
        generator = sg.mapper.GraphWaveGenerator(
            graph.value, scales=scales.as_dense(copy=False), degree=chebyshev_degree
        )
        nodes = graph.value.nodes().sort_values()
        embeddings_dataset = generator.flow(
            node_ids=nodes,
            sample_points=sample_points,
            batch_size=len(nodes),
            shuffle=False,
            repeat=False,
        )
        tf_tensor = list(embeddings_dataset)[0]  # TODO do we want a tensor matrix type?
        np_matrix = tf_tensor.numpy()
        matrix = NumpyMatrix(np_matrix)
        node2index = NumpyNodeMap(np.arange(len(nodes)), node_ids=nodes.to_numpy())
        return (matrix, node2index)

    @concrete_algorithm("embedding.train.graph_sage.mean")
    def sg_graph_sage_mean_train(
        graph: NetworkXGraph,
        node_features: NumpyMatrix,
        node2row: NumpyNodeMap,
        walk_length: int,
        walks_per_node: int,
        layer_sizes: NumpyVector,
        samples_per_layer: NumpyVector,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        dropout_probability: float = 0.0,
        use_bias: bool = True,
        normalization_method: str = "l2",
        worker_count: int = 1,
    ) -> StellarGraphGraphSageNodeEmbedding:
        assert set(graph.value.nodes()) == set(
            node2row.pos2id
        ), f"Nodes in graph {graph} do not match nodes features {node2row}"
        assert len(layer_sizes) == len(
            samples_per_layer
        ), f"Number of layer sizes ({len(layer_sizes)}) do not match the number of values specifiying the samples for each layer ({len(samples_per_layer)})"
        # StellarGraph cannot add node features after a graph has been created,
        # so it's less expensive to take a NetworkX graph and create a new StellarGraph
        # graph than to go from StellarGraph -> NetworkX -> StellarGraph
        node_features_df = pd.DataFrame(
            node_features.as_dense(copy=False), index=node2row.pos2id
        )
        sg_graph = sg.StellarGraph.from_networkx(
            graph.value,
            edge_weight_attr=graph.edge_weight_label or "weight",
            node_features=node_features_df,
        )

        # TODO samples_per_layer and layer_sizes are converted to lists; consider making a Python list or Tuple
        samples_per_layer_list = samples_per_layer.as_dense().tolist()
        unsupervised_samples = sg.data.UnsupervisedSampler(
            sg_graph,
            nodes=list(sg_graph.nodes()),
            length=walk_length,
            number_of_walks=walks_per_node,
        )
        generator = sg.mapper.GraphSAGELinkGenerator(
            sg_graph, batch_size, samples_per_layer_list
        )
        train_gen = generator.flow(unsupervised_samples)
        graphsage = sg.layer.GraphSAGE(
            layer_sizes=layer_sizes.as_dense().tolist(),
            generator=generator,
            bias=True,
            dropout=0.0,
            normalize="l2",
        )
        x_inp, x_out = graphsage.in_out_tensors()
        prediction = sg.layer.link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
        )(x_out)
        model = tf.keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[tf.keras.metrics.binary_accuracy],
        )
        with tempfile.TemporaryDirectory(
            prefix=f"stellargraph_graph_sage_{uuid.uuid4().hex}_"
        ) as tmp_dir:
            history = model.fit(
                train_gen,
                epochs=epochs,
                verbose=1,
                use_multiprocessing=False,
                workers=worker_count,
                shuffle=True,
                callbacks=tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(tmp_dir, "model.{epoch}.h5")
                ),
            )
            best_epoch_index = np.argmin(history.history["loss"])
            best_model_checkpoint_path = os.path.join(
                tmp_dir, f"model.{1+best_epoch_index}.h5"
            )
            model.load_weights(best_model_checkpoint_path)

        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = tf.keras.Model(inputs=x_inp_src, outputs=x_out_src)

        return StellarGraphGraphSageNodeEmbedding(
            embedding_model, samples_per_layer_list
        )
