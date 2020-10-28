from metagraph import translator
from metagraph.plugins import has_networkx
from .. import has_stellargraph

if has_networkx and has_stellargraph:
    import networkx as nx
    import stellargraph as sg
    import numpy as np
    import pandas as pd
    from .types import StellarGraph
    from metagraph.plugins.python.types import dtype_casting
    from metagraph.plugins.networkx.types import NetworkXGraph

    @translator
    def graph_networkx_to_stellargraph(x: NetworkXGraph, **props) -> StellarGraph:
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            x, {"node_dtype", "node_type", "edge_type"}
        )

        if aprops["node_type"] == "map":
            node_weight_index = 0
            nodes, weights = x.value.nodes(data=x.node_weight_label)
            weight_vectors = ([weight] for weight in weights)
            node_features = pd.DataFrame(weight_vectors, index=nodes)
        else:
            node_weight_index = None
            node_features = None

        is_weighted = aprops["edge_type"] == "map"

        if aprops["node_dtype"] == "int":
            dtype = np.int64  # TODO how much precision do we want?
        elif aprops["node_dtype"] == "float":
            dtype = np.float64  # TODO how much precision do we want?
        else:
            dtype = np.dtype("U")

        node_sg_type = "default"
        edge_sg_type = "default"
        sg_graph = sg.StellarGraph.from_networkx(
            x.value,
            edge_weight_attr=x.edge_weight_label,
            node_type_attr=node_sg_type,
            edge_type_attr=edge_sg_type,
            node_type_default=node_sg_type,
            edge_type_default=edge_sg_type,
            node_features=node_features,
            dtype=dtype,
        )
        return StellarGraph(
            sg_graph,
            node_weight_index=node_weight_index,
            is_weighted=is_weighted,
            node_sg_type=node_sg_type,
            edge_sg_type=edge_sg_type,
        )

    @translator
    def graph_stellargraph_to_networkx(x: StellarGraph, **props) -> NetworkXGraph:
        aprops = StellarGraph.Type.compute_abstract_properties(
            x, {"is_directed", "node_type", "node_dtype", "edge_type", "edge_dtype"}
        )
        nx_multi_graph = x.value.to_networkx(
            node_type_attr=x.node_sg_type,
            edge_type_attr=x.edge_sg_type,
            edge_weight_attr="weight",
            feature_attr="weight",
        )

        has_node_weights = aprops["node_type"] == "map"
        if has_node_weights:
            node_weight_label = "weight"
            caster = dtype_casting[aprops["node_dtype"]]
        else:
            node_weight_label = None
        for node, node_attributes in nx_multi_graph.nodes(data=True):
            del node_attributes[x.node_sg_type]
            if has_node_weights:
                node_attributes["weight"] = caster(
                    node_attributes["weight"][x.node_weight_index]
                )
            else:
                del node_attributes["weight"]

        is_weighted = aprops["edge_type"] == "map"
        if is_weighted:
            edge_weight_label = "weight"
            caster = dtype_casting[aprops["edge_dtype"]]
        else:
            edge_weight_label = None
        for src, dst, edge_attributes in nx_multi_graph.edges(data=True):
            del edge_attributes[x.edge_sg_type]
            if is_weighted:
                edge_attributes["weight"] = caster(edge_attributes["weight"])
            else:
                del edge_attributes["weight"]
        nx_graph = (
            nx.DiGraph(nx_multi_graph)
            if aprops["is_directed"]
            else nx.Graph(nx_multi_graph)
        )

        return NetworkXGraph(
            nx_graph,
            node_weight_label=node_weight_label,
            edge_weight_label=edge_weight_label,
        )
