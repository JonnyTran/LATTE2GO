import traceback
from abc import abstractmethod
from typing import List, Dict, Union, Tuple, Set, Any

import networkx as nx
import numpy as np
import openomics
import pandas as pd
from logzero import logger
from openomics.transforms.agg import concat_uniques
from sklearn import preprocessing

from moge.network.labels import select_labels, to_list_of_strs
SEQUENCE_COL = 'sequence'
MODALITY_COL = 'omic'
class Network(object):
    def __init__(self, networks: Dict[Tuple[str], nx.Graph]) -> None:
        """
        A class that manages multiple graphs and the nodes between those graphs. Inheriting this class will run .process_network() and get_node_list()
        :param networks (list): a list of Networkx Graph's
        """
        self.networks = networks
        self.process_network()
        self.node_list = self.get_all_nodes()

    def get_all_nodes(self) -> Set[str]:
        if isinstance(self.networks, dict):
            node_set = {node for network in self.networks.values() for node in network.nodes}
        elif isinstance(self.networks, list):
            node_set = {node for network in self.networks for node in network.nodes}
        else:
            node_set = {}

        return node_set

    def get_connected_nodes(self, layer: Union[str, Tuple[str, str, str]]):
        degrees = self.networks[layer].degree()
        return [node for node, deg in degrees if deg > 0]

    def remove_nodes_from(self, nodes: Union[List[str], Dict[str, Set[str]]]) -> None:
        nan_nodes = [node for node in self.get_all_nodes() \
                     if pd.isna(node) or type(node) != str or len(node) == 0]

        if isinstance(nodes, dict):
            # Ensure no empty lists
            nodes = {ntype: set(li) for ntype, li in nodes.items()}

            for metapath, g in self.networks.items():
                g_ntypes = {metapath[0], metapath[-1]}.intersection(nodes)

                if g_ntypes:
                    for ntype in g_ntypes:
                        g.remove_nodes_from(nodes[ntype])

                if nan_nodes:
                    g.remove_nodes_from(nan_nodes)

            for ntype, nodelist in nodes.items():
                if hasattr(self, 'nodes'):
                    self.nodes[ntype] = self.nodes[ntype].drop(nodelist)

                if hasattr(self, 'annotations'):
                    if hasattr(self, 'nodes'):
                        self.annotations[ntype] = self.annotations[ntype].loc[self.nodes[ntype]]
                    else:
                        self.annotations[ntype] = self.annotations[ntype].drop(nodelist)

        elif isinstance(nodes, list):
            for g in self.networks.values() if isinstance(self.networks, dict) else self.networks:
                g.remove_nodes_from(nodes)
                if nan_nodes:
                    g.remove_nodes_from(nan_nodes)

    @abstractmethod
    def process_network(self):
        raise NotImplementedError

    @abstractmethod
    def add_nodes(self, nodes: List[str], ntype: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_edges(self, edgelist: List[Union[Tuple[str, str], Tuple[str, str, Dict[str, Any]]]], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def import_edgelist_file(self, filepath, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_edgelist(self, node_list: List[str], inclusive=True, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_adjacency_matrix(self, edge_types: List, node_list=None, method="GAT", output="csr", **kwargs):
        """
        Retrieves the adjacency matrix of a subnetwork with `edge_types` edge types  and `node_list` nodes. The adjacency
        matrix is preprocessed for `method`, e.g. adding self-loops in GAT, and is converted to a sparse matrix of `output` type.

        :param edge_types: a list of edge types to retrieve.
        :param node_list: list of nodes.
        :param method: one of {"GAT", "GCN"}, default: "GAT".
        :param output: one of {"csr", "coo", "dense"}, default "csr":
        :param kwargs:
        """
        raise NotImplementedError

    @abstractmethod
    def get_graph_laplacian(self, edge_types: List, node_list=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, i, j, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def remove_edges_from(self, edgelist, **kwargs):
        raise NotImplementedError

    def slice_adj(self, adj, nodes_A, nodes_B=None):
        if nodes_B is None:
            idx = [self.node_list.index(node) for node in nodes_A]
            return adj[idx, :][:, idx]
        else:
            idx_A = [self.node_list.index(node) for node in nodes_A]
            idx_B = [self.node_list.index(node) for node in nodes_B]
            return adj[idx_A, :][:, idx_B]


class AttributedNetwork(Network):
    def __init__(self, multiomics: openomics.MultiOmics, annotations=True, **kwargs) -> None:
        """
        Handles the MultiOmics attributes associated to the network(s).

        :param multiomics: an openomics.MultiOmics instance.
        :param annotations: default True. Whether to run annotations processing.
        :param kwargs: args to pass to Network() constructor.
        """
        self.multiomics = multiomics

        # Process network & node_list
        super().__init__(**kwargs)

        # Process node attributes
        if annotations:
            self.process_annotations()
            self.process_feature_tranformer()

    def process_annotations(self):
        annotations_list = []

        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            annotation[MODALITY_COL] = modality
            annotations_list.append(annotation)

        self.annotations = pd.concat(annotations_list, join="inner", copy=True)
        assert type(
            self.annotations.index) != pd.MultiIndex, "Annotation index must be a pandas.Index type and not a MultiIndex."

        # self.annotations = self.annotations[~self.annotations.index.duplicated(keep='first')]
        self.annotations = self.annotations.groupby(self.annotations.index).agg(
            {k: concat_uniques for k in self.annotations.columns})

        print("Annotation columns:", self.annotations.columns.tolist())

    def process_feature_tranformer(self, columns=None, delimiter="\||;", labels_subset=None, min_count=0,
                                   verbose=False):
        """
        For each of the annotation column, create a sklearn label binarizer. If the column data is delimited, a MultiLabelBinarizer
        is used to convert a list of labels into a vector.
        :param delimiter (str): default "|".
        :param min_count (int): default 0. Remove labels with frequency less than this. Used for classification or train/test stratification tasks.

        Args:
            columns ():
        """
        self.delimiter = delimiter

        if not hasattr(self, "feature_transformer"):
            self.feature_transformer = {}

        df = self.annotations
        if columns:
            df.filter(columns, axis='columns')
        transformers = self.get_feature_transformers(df, node_list=self.node_list, labels_subset=labels_subset,
                                                     min_count=min_count,
                                                     delimiter=delimiter, verbose=verbose)
        self.feature_transformer.update(transformers)

    @classmethod
    def get_feature_transformers(cls, annotation: pd.DataFrame,
                                 labels_subset: List[str] = None,
                                 min_count: int = 0,
                                 delimiter="\||;",
                                 verbose=False) \
            -> Dict[str, Union[preprocessing.MultiLabelBinarizer, preprocessing.StandardScaler]]:
        """
        :param annotation: a pandas DataFrame
        :param node_list: list of nodes. Indexes the annotation DataFrame
        :param labels_subset: str or list of str for the labels to filter by min_count
        :param min_count: minimum frequency of label to keep
        :param delimiter: default "\||;", delimiter ('|' or ';') to split strings
        :return: dict of feature transformers
        """
        transformers: Dict[str, preprocessing.MultiLabelBinarizer] = {}
        for col in annotation.columns:
            if col == SEQUENCE_COL:
                continue

            values: pd.Series = annotation[col].dropna(axis=0)
            if values.map(type).nunique() > 1:
                logger.warn(f"{col} has more than 1 dtypes: {values.map(type).unique()}")

            try:
                if annotation[col].dtypes == np.object and (annotation[col].dropna().map(type) == str).all():
                    transformers[col] = preprocessing.MultiLabelBinarizer()

                    if annotation[col].str.contains(delimiter, regex=True).any():
                        logger.info("Label {} (of str split by '{}') transformed by MultiLabelBinarizer".format(col,
                                                                                                                delimiter)) if verbose else None
                        values = values.str.split(delimiter)
                        values = values.map(
                            lambda x: [term.strip() for term in x if len(term) > 0] if isinstance(x, list) else x)

                    if labels_subset is not None and col in labels_subset and min_count:
                        labels_subset = select_labels(values, min_count=min_count)
                        values = values.map(lambda labels: [item for item in labels if item not in labels_subset])

                    transformers[col].fit(values)

                elif annotation[col].dtypes == int or annotation[col].dtypes == float:
                    logger.info("Label {} (of int/float) is transformed by StandardScaler".format(col)) \
                        if verbose else None
                    transformers[col] = preprocessing.StandardScaler()

                    values = values.dropna().to_numpy()
                    transformers[col].fit(values.reshape(-1, 1))

                else:
                    logger.info("Label {} is transformed by MultiLabelBinarizer".format(col)) if verbose else None
                    transformers[col] = preprocessing.MultiLabelBinarizer()
                    values = values.map(to_list_of_strs)

                    transformers[col].fit(values)

                if hasattr(transformers[col], 'classes_') and \
                        ("" in transformers[col].classes_ or pd.isna(transformers[col].classes_).any()):
                    logger.warn(f"removed '' from classes in {col}")
                    transformers[col].classes_ = np.delete(transformers[col].classes_,
                                                           np.where(transformers[col].classes_ == "")[0])
            except Exception as e:
                logger.error(f"`{col}` dtypes: {values.map(type).unique()}, {e.__class__}: {e}")
                print(traceback.format_exc())
                continue

            logger.info(f'get_feature_transformers `{col}`: '
                        f'{transformers[col].classes_.shape if hasattr(transformers[col], "classes_") else ""}, '
                        f'min_count: {min_count}')

        return transformers

    def get_labels_color(self, label, go_id_colors, child_terms=True, fillna="#e5ecf6", label_filter=None):
        """
        Filter the gene GO annotations and assign a color for each term given :param go_id_colors:.
        """
        if hasattr(self, "all_annotations"):
            labels = self.all_annotations[label].copy(deep=True)
        else:
            labels = self.annotations[label].copy(deep=True)

        if labels.str.contains("\||;", regex=True).any():
            labels = labels.str.split("\||;")

        if label_filter is not None:
            # Filter only annotations in label_filter
            if not isinstance(label_filter, set): label_filter = set(label_filter)
            labels = labels.map(lambda x: [term for term in x if term in label_filter] if x and len(x) > 0 else None)

        # Filter only annotations with an associated color
        labels = labels.map(lambda x: [term for term in x if term in go_id_colors.index] if x and len(x) > 0 else None)

        # For each node select one term
        labels = labels.map(lambda x: sorted(x)[-1 if child_terms else 0] if x and len(x) >= 1 else None)
        label_color = labels.map(go_id_colors)
        if fillna:
            label_color.fillna("#e5ecf6", inplace=True)
        return label_color
