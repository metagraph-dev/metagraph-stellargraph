from metagraph.types import Graph, GraphSageNodeEmbedding
from metagraph.wrappers import GraphWrapper, GraphSageNodeEmbeddingWrapper
from .. import has_stellargraph
from typing import Set, List, Dict, Any
import copy
import math
import numpy as np


def _determine_array_dtype(array: np.ndarray) -> str:
    if array.dtype.char in np.typecodes["AllFloat"]:
        dtype_str = "float"
    elif array.dtype.char == np.dtype("bool"):
        dtype_str = "bool"
    elif array.dtype.char in np.typecodes["AllInteger"]:
        dtype_str = "int"
    else:
        dtype_str = "str"
    return dtype_str


if has_stellargraph:
    import stellargraph as sg
    import tensorflow as tf

    class StellarGraph(GraphWrapper, abstract=Graph):
        def __init__(
            self,
            sg_graph,
            node_weight_index=None,
            is_weighted=False,
            node_sg_type="default",
            edge_sg_type="default",
        ):
            super().__init__()
            # TODO can we use ilocs for better performance? https://stellargraph.readthedocs.io/en/stable/api.html?highlight=StellarGraph.edges#iloc-explanation
            self._assert_instance(sg_graph, sg.StellarGraph)
            self._assert(
                node_sg_type in sg_graph.node_types,
                f"Unknown stellargraph node type {node_sg_type}",
            )
            self._assert(
                edge_sg_type in sg_graph.edge_types,
                f"Unknown stellargraph edge type {edge_sg_type}",
            )
            if node_weight_index is not None:
                self._assert_instance(node_weight_index, int)
                valid_index_range_end = (
                    sg_graph.node_features(node_type=node_sg_type, nodes=[]).shape[1]
                    - 1
                )
                self._assert(
                    0 <= node_weight_index < valid_index_range_end,
                    f"node weight index ({node_weight_index})is not valid.",
                )
            self.value = sg_graph
            self.node_weight_index = node_weight_index
            self.is_weighted = is_weighted
            self.node_sg_type = node_sg_type
            self.edge_sg_type = edge_sg_type

        def copy(self):
            return StellarGraph(
                copy.deepcopy(self.value),
                self.node_weight_index,
                self.is_weighted,
                self.node_sg_type,
                self.edge_sg_type,
            )

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {
                    "is_directed",
                    "node_type",
                    "node_dtype",
                    "edge_type",
                    "edge_dtype",
                } - ret.keys():

                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()
                    if prop == "node_type":
                        ret[prop] = "set" if obj.node_weight_index is None else "map"
                    if prop == "node_dtype":
                        ret[prop] = _determine_array_dtype(obj.value.node_features())
                    if prop == "edge_type":
                        ret[prop] = "set" if obj.is_weighted is None else "map"
                    if prop == "edge_dtype":
                        ret[prop] = _determine_array_dtype(
                            obj.value.edge_arrays(include_edge_weight=True)[3]
                        )

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                if {"edge_has_negative_weights"} & slow_props:
                    _, weights = obj.value.edges(include_edge_weight=True)
                    ret[prop] = (weights < 0).any()

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                g1 = obj1.value
                g2 = obj2.value

                # Compare

                g1_nodes = g1.nodes(node_type=obj1.node_sg_type)
                g2_nodes = g2.nodes(node_type=obj2.node_sg_type)
                assert set(g1_nodes) == set(
                    g2_nodes
                ), f"{set(g1_nodes)} != {set(g2_nodes)}"
                if aprops1.get("node_type") == "map":
                    g1_node_weights = g1.node_features(
                        node_type=obj1.node_sg_type, nodes=g1_nodes
                    )[:, obj1.node_weight_index]
                    g2_node_weights = g2.node_features(
                        node_type=obj2.node_sg_type, nodes=g1_nodes
                    )[:, obj2.node_weight_index]
                    if aprops1["node_dtype"] == "float":
                        assert np.isclose(
                            g1_node_weights, g2_node_weights, rtol=rel_tol, atol=abs_tol
                        ).all(), f"{g1_node_weights} != {g2_node_weights}"
                    else:
                        assert (
                            g1_node_weights == g2_node_weights
                        ).all(), f"{g1_node_weights} != {g2_node_weights}"

                is_directed = aprops1.get("is_directed")
                is_weighted = aprops1.get("edge_type") == "map"

                def get_obj_edge_set(obj):
                    g = obj.value
                    edges = zip(
                        *g.edge_arrays(include_edge_type=True, include_edge_weight=True)
                    )
                    edges = filter(lambda edge: edge[2] == obj.edge_sg_type, edges)
                    canonicalize_edge = (
                        lambda edge: edge
                        if is_directed
                        else tuple(sorted(edge[:2])) + (edges[2:],)
                    )
                    edges = map(canonicalize_edge, edges)
                    edges = {(src, dst): weight for src, dst, _, weight in edges}
                    return edges

                g1_edges = get_obj_edge_set(obj1)
                g2_edges = get_obj_edge_set(obj2)

                assert set(g1_edges.keys()) == set(
                    g2_edges.keys()
                ), f"{set(g1_edges.keys())} != {set(g2_edges.keys())}"

                if is_weighted:
                    edge_weights_are_floats = aprops1["edge_dtype"] == "float"
                    for edge, g1_weight in g1_edges.items():
                        g2_weight = g1_edges[edge]
                        if edge_weights_are_floats:
                            assert math.isclose(
                                g1_weight, g2_weight, rel_tol=rel_tol, abs_tol=abs_tol
                            ), f"Weights differ for edge {edge}"
                        else:
                            assert (
                                g1_weight == g2_weight
                            ), f"Weights differ for edge {edge}"

    class StellarGraphGraphSageNodeEmbedding(
        GraphSageNodeEmbeddingWrapper, abstract=GraphSageNodeEmbedding
    ):
        def __init__(self, model, samples_per_layer: List[int]):
            super().__init__()
            self._assert_instance(model, tf.keras.Model)
            self._assert_instance(samples_per_layer, list)
            self._assert(
                all(
                    isinstance(sample_count, int) for sample_count in samples_per_layer
                ),
                f"{samples_per_layer} is not a list of int values",
            )
            self.model = model
            self.samples_per_layer = samples_per_layer  # TODO this could be a dynamic value given at embeddding time

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()
                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                assert False, f"Cannot compare instances of {cls.__qualname__}"
