from .. import has_stellargraph
from metagraph import concrete_algorithm

if has_stellargraph:
    import stellargraph as sg
    from metagraph.plugins.python.types import PythonNodeMap, PythonNodeSet
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
        return PythonNodeMap(index_to_label,)
