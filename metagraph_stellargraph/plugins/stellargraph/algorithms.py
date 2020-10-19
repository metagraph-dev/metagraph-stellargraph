from .. import has_stellargraph
from metagraph import concrete_algorithm

if has_stellargraph:
    import stellargraph as sg
    import tensorflow as tf
    import numpy as np
    from metagraph.plugins.python.types import PythonNodeMap, PythonNodeSet
    from metagraph.plugins.numpy.types import (
        NumpyNodeMap,
        NumpyMatrix,
        NumpyNodeEmbedding,
    )
    from .types import StellarGraph

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
    ) -> NumpyNodeEmbedding:

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
            metrics=[tf.keras.metrics.binary_accuracy],  # @todo comment this out
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
        return NumpyNodeEmbedding(matrix, node2index)
