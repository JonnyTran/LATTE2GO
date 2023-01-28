from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Dict, Mapping, Any, Tuple

import dgl
import networkx as nx
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.nn.inits import glorot, zeros
from transformers import BertForSequenceClassification, BertConfig

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset


class LabelNodeClassifer(nn.Module):
    def __init__(self, dataset: HeteroNodeClfDataset, hparams: Namespace):
        """
        Compute node classification scores for each node against each class in the graph using DistMult. The classes
        may be a subset of the nodes from different node types.
        Args:
            dataset ():
            hparams ():
        """
        super().__init__()
        self.n_classes = hparams.n_classes
        self.classes = dataset.classes
        self.head_node_type = hparams.head_node_type

        self.pred_ntypes = tuple(dataset.pred_ntypes)
        assert isinstance(self.pred_ntypes, (list, tuple)), f'self.pred_ntypes = {self.pred_ntypes}'

        self.class_indices = dataset.class_indices
        assert self.class_indices, f'self.class_indices ({self.class_indices}) must not be none'
        self.class_sizes = {ntype: idx[idx != -1].numel() \
                            for ntype, idx in self.class_indices.items()}

        # if hparams.embedding_dim
        self.embedding_dim = hparams.embedding_dim
        self.weights = nn.ParameterDict({ntype: torch.rand(hparams.embedding_dim) for ntype in self.pred_ntypes})
        # self.bias = nn.ParameterDict({ntype: torch.zeros(self.n_classes) for ntype in self.pred_ntypes})

        if hparams.loss_type == "BCE":
            self.activation = nn.Sigmoid()
        elif hparams.loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.activation = nn.Softmax()

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'weight'):
            for ntype in self.weights:
                nn.init.xavier_uniform_(self.weights[ntype])

    def forward(self, embeddings: Tensor, h_dict: Dict[str, Tensor], **kwargs) -> Tensor:
        cls_logits = {}
        for ntype in self.pred_ntypes:
            # First n indices in `h_dict[ntype]` are class nodes
            cls_emb = h_dict[ntype][:self.class_sizes[ntype]]
            cls_logits[ntype] = ((embeddings * self.weights[ntype]) @ cls_emb.T)  # + self.bias[ntype]

        logits = torch.cat([cls_logits[ntype] for ntype in self.pred_ntypes], dim=1)
        reorder_indices = self.get_reorder_indices(self.class_sizes, self.class_indices, pred_ntypes=self.pred_ntypes)
        logits = logits[:, reorder_indices]

        if hasattr(self, 'activation'):
            logits = self.activation(logits)
        return logits

    def get_reorder_indices(self, class_sizes: Dict[str, int], class_indices: Dict[str, Tensor],
                            pred_ntypes: Tuple[str]) \
            -> Tensor:
        """
        Return a permutation of the indices among the classes, which contain a combined mix of classes of different
        `pred_ntypes`.
        Args:
            class_sizes ():
            class_indices ():
            pred_ntypes ():

        Returns:
            reorder_idx: a permutation of range(n_classes) where applied to `logits`,
        """
        if hasattr(self, 'reorder_idx') and isinstance(self.reorder_idx, Tensor):
            return self.reorder_idx

        mix_indices = {k: (v != -1).nonzero().flatten() for k, v in class_indices.items()}
        cls_idx = torch.stack([class_indices[ntype] for ntype in pred_ntypes], axis=0)

        _, sort_indices = cls_idx.sort(dim=0, descending=True)

        class_idx_offset = {}
        offset = 0
        for i, ntype in enumerate(pred_ntypes):
            class_idx_offset[ntype] = offset
            offset += class_sizes[ntype]

        reorder_idx = []
        for i in range(self.n_classes):
            ntype = pred_ntypes[sort_indices[0][i]]
            cls_offset = class_idx_offset[ntype]
            print((mix_indices[ntype] == i).nonzero().flatten())
            cls_idx = cls_offset + (mix_indices[ntype] == i).nonzero().flatten().item()
            reorder_idx.append(cls_idx)

        reorder_idx = torch.tensor(reorder_idx, dtype=torch.int64)
        self.reorder_idx = reorder_idx

        return reorder_idx


class DenseClassification(nn.Module):
    def __init__(self, hparams: Namespace):
        super().__init__()
        # Classifier
        if getattr(hparams, 'nb_cls_dense_size', 0) > 0:
            self.linears = nn.Sequential(OrderedDict([
                ("linear_1", nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=getattr(hparams, 'nb_cls_dropout', 0.0))),
                ("linear", nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes))
            ]))
        else:
            self.linears = nn.Sequential(OrderedDict([
                ("linear", nn.Linear(hparams.embedding_dim, hparams.n_classes))
            ]))

        # Activation
        self.loss_type = hparams.loss_type
        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            print("INFO: Output of `classifier` is logits")

        elif "NEGATIVE_LOG_LIKELIHOOD" == self.loss_type:
            print("INFO: Output of `classifier` is LogSoftmax")
            self.linears.add_module("activation", nn.LogSoftmax(dim=1))

        elif "SOFTMAX_CROSS_ENTROPY" == self.loss_type:
            print("INFO: Output of `classifier` is logits")

        elif "BCE" == self.loss_type:
            print("INFO: Output of `classifier` is sigmoid probabilities")
            self.linears.add_module("activation", nn.Sigmoid())

        else:
            print("INFO: [Else Case] Output of `classifier` is logits")

        self.reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def reset_parameters(self):
        for linear in self.linears:
            if hasattr(linear, "weight"):
                nn.init.xavier_uniform_(linear.weight)

    def forward(self, h, **kwargs):
        h = self.linears(h)

        return h


