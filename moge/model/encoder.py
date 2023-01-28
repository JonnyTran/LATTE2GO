import logging
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, Union, List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_sparse import SparseTensor
from transformers import BertConfig, BertForSequenceClassification

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.tensor import tensor_sizes

logging.getLogger("transformers").setLevel(logging.ERROR)


class HeteroNodeFeatureEncoder(nn.Module):
    def __init__(self, hparams: Namespace, dataset: HeteroGraphDataset, subset_ntypes: List[str] = None) -> None:
        super().__init__()
        self.embeddings = self.init_embeddings(embedding_dim=hparams.embedding_dim,
                                               num_nodes_dict=dataset.num_nodes_dict,
                                               in_channels_dict=dataset.node_attr_shape,
                                               pretrain_embeddings=hparams.node_emb_init if "node_emb_init" in hparams else None,
                                               freeze=hparams.freeze_embeddings if "freeze_embeddings" in hparams else True,
                                               subset_ntypes=subset_ntypes, )
        print("model.encoder.embeddings: ", tensor_sizes(self.embeddings))

        # node types that needs a projection to align to the embedding_dim
        linear_dict = {}
        for ntype in dataset.node_types:
            if ntype in dataset.node_attr_shape and dataset.node_attr_shape[ntype] and \
                    dataset.node_attr_shape[ntype] != hparams.embedding_dim:

                # Row Sparse Matrix Multiplication
                if ntype in dataset.node_attr_sparse:
                    linear_dict[ntype] = nn.ParameterList([
                        nn.EmbeddingBag(dataset.node_attr_shape[ntype], hparams.embedding_dim,
                                        mode='sum', include_last_offset=True),
                        nn.Parameter(torch.zeros(hparams.embedding_dim)),
                        nn.ReLU(),
                        nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0)
                    ])

                else:
                    linear_dict[ntype] = nn.Sequential(OrderedDict(
                        [('linear', nn.Linear(dataset.node_attr_shape[ntype], hparams.embedding_dim)),
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0))]))

            elif ntype in self.embeddings and self.embeddings[ntype].weight.size(1) != hparams.embedding_dim:
                # Pretrained embeddings size doesn't match embedding_dim
                linear_dict[ntype] = nn.Sequential(OrderedDict(
                    [('linear', nn.Linear(self.embeddings[ntype].weight.size(1), hparams.embedding_dim)),
                     ('relu', nn.ReLU()),
                     ('dropout', nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0))]))

        self.linear_proj = nn.ModuleDict(linear_dict)
        print("model.encoder.feature_projection: ", self.linear_proj)

        self.reset_parameters()

    def reset_parameters(self):
        for ntype, linear in self.linear_proj.items():
            if hasattr(linear, "weight"):
                nn.init.xavier_uniform_(linear.weight)

        for ntype, embedding in self.embeddings.items():
            if hasattr(embedding, "weight"):
                nn.init.xavier_uniform_(embedding.weight)

    def init_embeddings(self, embedding_dim: int,
                        num_nodes_dict: Dict[str, int],
                        in_channels_dict: Dict[str, int],
                        pretrain_embeddings: Dict[str, Tensor],
                        subset_ntypes: List[str] = None,
                        freeze=True) -> Dict[str, nn.Embedding]:
        # If some node type are not attributed, instantiate nn.Embedding for them
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []

        if subset_ntypes:
            non_attr_node_types = set(ntype for ntype in non_attr_node_types if ntype in subset_ntypes)

        embeddings = {}
        for ntype in non_attr_node_types:
            if pretrain_embeddings is None or ntype not in pretrain_embeddings:
                print("Initialized trainable embeddings: ", ntype)
                embeddings[ntype] = nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                                 embedding_dim=embedding_dim,
                                                 scale_grad_by_freq=True,
                                                 sparse=False)

                nn.init.xavier_uniform_(embeddings[ntype].weight)

            else:
                print(f"Pretrained embeddings freeze={freeze}", ntype)
                max_norm = pretrain_embeddings[ntype].norm(dim=1).mean()
                embeddings[ntype] = nn.Embedding.from_pretrained(pretrain_embeddings[ntype],
                                                                 freeze=freeze,
                                                                 scale_grad_by_freq=True,
                                                                 max_norm=max_norm)

        embeddings = nn.ModuleDict(embeddings)

        return embeddings

    def forward(self, feats: Dict[str, Tensor], global_node_index: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h_dict = {k: v for k, v in feats.items()} if isinstance(feats, dict) else {}

        for ntype in global_node_index:
            if global_node_index[ntype].numel() == 0: continue

            if ntype not in h_dict and ntype in self.embeddings:
                h_dict[ntype] = self.embeddings[ntype](global_node_index[ntype]).to(global_node_index[ntype].device)

            # project to embedding_dim if node features are of not the same dimension
            if ntype in self.linear_proj and isinstance(h_dict[ntype], Tensor):
                if hasattr(self, "batchnorm"):
                    h_dict[ntype] = self.batchnorm[ntype].forward(h_dict[ntype])

                h_dict[ntype] = self.linear_proj[ntype].forward(h_dict[ntype])

            # Sparse matrix mult
            elif ntype in self.linear_proj and isinstance(h_dict[ntype], SparseTensor):
                self.linear_proj[ntype]: nn.ParameterList
                for module in self.linear_proj[ntype]:
                    if isinstance(module, nn.EmbeddingBag):
                        sparse_tensor: SparseTensor = h_dict[ntype]
                        h_dict[ntype] = module.forward(input=sparse_tensor.storage.col(),
                                                       offsets=sparse_tensor.storage.rowptr(),
                                                       per_sample_weights=sparse_tensor.storage.value())
                    elif isinstance(module, (nn.Parameter, Tensor)):
                        h_dict[ntype] = h_dict[ntype] + module
                    elif isinstance(module, nn.ReLU):
                        h_dict[ntype] = module.forward(h_dict[ntype])
                    elif isinstance(module, nn.Dropout):
                        h_dict[ntype] = module.forward(h_dict[ntype])

        return h_dict

