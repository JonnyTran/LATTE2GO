import logging
import math
import traceback
from argparse import Namespace
from collections import defaultdict
from typing import Dict, Iterable, Union, Tuple, List

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_sparse.sample
import tqdm
from fairscale.nn import auto_wrap
from logzero import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import MetaPath2Vec as Metapath2vec
from torch_sparse import SparseTensor

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.PyG.conv import HGT, RGCN
from moge.model.PyG.latte import LATTE as LATTE_Flat
from moge.model.PyG.metapaths import get_edge_index_values
from moge.model.PyG.relations import RelationAttention, RelationMultiLayerAgg
from moge.model.classifier import DenseClassification, LabelNodeClassifer
from moge.model.dgl.DeepGraphGO import pair_aupr, fmax
from moge.model.encoder import HeteroSequenceEncoder, HeteroNodeFeatureEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.tensor import filter_samples_weights, stack_tensor_dicts, activation, concat_dict_batch, to_device
from moge.model.trainer import NodeClfTrainer, print_pred_class_counts


class AFPNodeClf(NodeClfTrainer):
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        if hasattr(self, "feature_projection"):
            for ntype in self.feature_projection:
                nn.init.xavier_normal_(self.feature_projection[ntype].weights, gain=gain)

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        if hasattr(self, "seq_encoder"):
            self.seq_encoder = auto_wrap(self.seq_encoder)
        if hasattr(self, "encoder"):
            self.encoder = auto_wrap(self.encoder)


    def on_validation_epoch_start(self) -> None:
        if hasattr(self.embedder, 'layers'):
            for l, layer in enumerate(self.embedder.layers):
                if isinstance(layer, RelationAttention):
                    layer.reset()
        super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        if hasattr(self.embedder, 'layers'):
            for l, layer in enumerate(self.embedder.layers):
                if isinstance(layer, RelationAttention):
                    layer.reset()
        super().on_test_epoch_start()

    def on_predict_epoch_start(self) -> None:
        if hasattr(self.embedder, 'layers'):
            for l, layer in enumerate(self.embedder.layers):
                if isinstance(layer, RelationAttention):
                    layer.reset()
        super().on_predict_epoch_start()

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        scores = self.forward(X)

        scores, y_true, weights = stack_tensor_dicts(scores, y_true, weights)
        scores, y_true, weights = filter_samples_weights(y_pred=scores, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        loss = self.criterion.forward(scores, y_true, weights=weights)
        self.update_node_clf_metrics(self.train_metrics, scores, y_true, weights)

        if batch_nb % 100 == 0 and isinstance(self.train_metrics, Metrics):
            logs = self.train_metrics.compute_metrics()
        else:
            logs = {}

        self.log("loss", loss, on_step=True)
        self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)
        self.log("test_loss", test_loss)

        return test_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        predict_loss = self.criterion(y_pred, y_true)
        self.test_metrics.update_metrics(y_pred, y_true)

        self.log("predict_loss", predict_loss)

        return predict_loss

    @torch.no_grad()
    def predict(self, dataloader: DataLoader, node_names: pd.Index = None, save_betas=False, **kwargs) \
            -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
        """

        Args:
            dataloader ():
            node_names (): an pd.Index or np.ndarray to map node indices to node names for `head_node_type`.
            **kwargs ():

        Returns:
            targets, scores, embeddings, ntype_nids
        """
        if isinstance(self.embedder, RelationMultiLayerAgg):
            self.embedder.reset()

        head_ntype = self.head_node_type

        y_trues = []
        y_preds = []
        ntype_embs = defaultdict(lambda: [])
        ntype_nids = defaultdict(lambda: [])

        for batch in tqdm.tqdm(dataloader, desc='Predict dataloader'):
            X, y_true, weights = to_device(batch, device=self.device)
            h_dict, logits = self.forward(X, save_betas=save_betas, return_embeddings=True)
            y_pred = activation(logits, loss_type=self.hparams['loss_type'])

            y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
            y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
            idx = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights, return_index=True)

            y_true = y_true[idx]
            y_pred = y_pred[idx]

            global_node_index = X["global_node_index"][-1] \
                if isinstance(X["global_node_index"], (list, tuple)) else X["global_node_index"]

            # Convert all to CPU device
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            # Select index and add embeddings and nids
            for ntype, emb in h_dict.items():
                nid = global_node_index[ntype]
                if ntype == head_ntype:
                    emb = emb[idx]
                    nid = nid[idx]

                ntype_embs[ntype].append(emb.cpu().numpy())
                ntype_nids[ntype].append(nid.cpu().numpy())

        # Concat all batches
        targets = np.concatenate(y_trues, axis=0)
        scores = np.concatenate(y_preds, axis=0)
        ntype_embs = {ntype: np.concatenate(emb, axis=0) for ntype, emb in ntype_embs.items()}
        ntype_nids = {ntype: np.concatenate(nid, axis=0) for ntype, nid in ntype_nids.items()}

        if node_names is not None:
            index = node_names[ntype_nids[head_ntype]]
        else:
            index = pd.Index(ntype_nids[head_ntype], name="nid")

        targets = pd.DataFrame(targets, index=index, columns=self.dataset.classes)
        scores = pd.DataFrame(scores, index=index, columns=self.dataset.classes)
        ntype_embs = {ntype: pd.DataFrame(emb, index=ntype_nids[ntype]) for ntype, emb in ntype_embs.items()}

        return targets, scores, ntype_embs, ntype_nids

    def on_validation_end(self) -> None:
        super().on_validation_end()

        self.log_relation_atten_values()

    def on_test_end(self):
        super().on_test_end()

        try:
            if self.wandb_experiment is not None:
                self.eval()
                y_true, scores, embeddings, global_node_index = self.predict(self.test_dataloader(), save_betas=3)

                if hasattr(self.dataset, "nodes_namespace"):
                    y_true_dict = self.dataset.split_array_by_namespace(y_true, axis=1)
                    y_pred_dict = self.dataset.split_array_by_namespace(scores, axis=1)

                    for namespace in y_true_dict.keys():
                        if self.head_node_type in self.dataset.nodes_namespace:
                            # nids = global_node_index[self.head_node_type]
                            nids = y_true_dict[namespace].index
                            split_samples = self.dataset.nodes_namespace[self.head_node_type].iloc[nids]
                            title = f"{namespace}_PR_Curve_{split_samples.name}"
                        else:
                            split_samples = None
                            title = f"{namespace}_PR_Curve"

                        # Log AUPR and FMax for whole test set
                        if any(('fmax' in metric or 'aupr' in metric) for metric in self.test_metrics.metrics.keys()):
                            final_metrics = {
                                f'test_{namespace}_aupr': pair_aupr(y_true_dict[namespace], y_pred_dict[namespace]),
                                f'test_{namespace}_fmax': fmax(y_true_dict[namespace], y_pred_dict[namespace])[0], }
                            logger.info(f"final metrics {final_metrics}")
                            self.wandb_experiment.log(final_metrics | {'epoch': self.current_epoch + 1})

                        self.plot_pr_curve(targets=y_true_dict[namespace], scores=y_pred_dict[namespace],
                                           split_samples=split_samples, title=title)

                self.log_relation_atten_values()

                self.plot_embeddings_tsne(global_node_index=global_node_index,
                                          embeddings=embeddings,
                                          targets=y_true, y_pred=scores)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

    def train_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        if 't_order' in self.hparams and self.hparams.t_order > 1 and isinstance(self.embedder, RelationMultiLayerAgg):
            kwargs['add_metapaths'] = self.embedder.get_metapaths_chain()
        return super().train_dataloader(batch_size, num_workers, **kwargs)

    def val_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        if 't_order' in self.hparams and self.hparams.t_order > 1 and isinstance(self.embedder, RelationMultiLayerAgg):
            kwargs['add_metapaths'] = self.embedder.get_metapaths_chain()
        return super().val_dataloader(batch_size, num_workers, **kwargs)

    def test_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        if 't_order' in self.hparams and self.hparams.t_order > 1 and isinstance(self.embedder, RelationMultiLayerAgg):
            kwargs['add_metapaths'] = self.embedder.get_metapaths_chain()
        return super().test_dataloader(batch_size, num_workers, **kwargs)


class LATTEFlatNodeClf(AFPNodeClf):
    dataset: HeteroNodeClfDataset

    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super(AFPNodeClf, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.n_layers}-{hparams.t_order}"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(self.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        if dataset.pred_ntypes is not None:
            hparams.pred_ntypes = dataset.pred_ntypes

        self.embedder = LATTE_Flat(n_layers=hparams.n_layers,
                                   t_order=hparams.t_order,
                                   embedding_dim=hparams.embedding_dim,
                                   num_nodes_dict=dataset.num_nodes_dict,
                                   metapaths=dataset.get_metapaths(),
                                   layer_pooling=hparams.layer_pooling,
                                   activation=hparams.activation,
                                   attn_heads=hparams.attn_heads,
                                   attn_activation=hparams.attn_activation,
                                   attn_dropout=hparams.attn_dropout,
                                   edge_sampling=getattr(hparams, 'edge_sampling', False),
                                   hparams=hparams)

        # Output layer
        if self.embedder.layer_pooling == 'concat':
            hparams.embedding_dim = hparams.embedding_dim * hparams.n_layers
        elif dataset.pred_ntypes is not None and dataset.class_indices:
            self.classifier = LabelNodeClassifer(dataset, hparams)
        else:
            self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(
            loss_type=hparams.loss_type, n_classes=dataset.n_classes,
            class_weight=getattr(dataset, 'class_weight', None) \
                if getattr(hparams, 'use_class_weights', False) else None,
            pos_weight=getattr(dataset, 'pos_weight', None) \
                if getattr(hparams, 'use_pos_weights', False) else None,
            multilabel=dataset.multilabel,
            reduction=getattr(hparams, 'reduction', 'mean'))



    def forward(self, inputs: Dict[str, Union[Tensor, Dict[Union[str, Tuple[str]], Union[Tensor, int]]]],
                return_embeddings=False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        batch_sizes = inputs["batch_size"]
        if isinstance(self.classifier, LabelNodeClassifer):
            # Ensure we count betas and alpha values for the `pred_ntypes`
            batch_sizes = batch_sizes | self.classifier.class_sizes

        h_out = {}
        #  Node feature or embedding input
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  split_size=math.sqrt(self.hparams.batch_size // 4)))
        # Node sequence features
        if len(h_out) < len(inputs["global_node_index"].keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_index=inputs["global_node_index"])
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        # GNN embedding
        h_out = self.embedder.forward(h_out,
                                      edge_index_dict=inputs["edge_index_dict"],
                                      global_node_index=inputs["global_node_index"],
                                      batch_sizes=batch_sizes,
                                      **kwargs)

        # Node classification
        if hasattr(self, "classifier"):
            head_ntype_embeddings = h_out[self.head_node_type]
            if "batch_size" in inputs and self.head_node_type in batch_sizes:
                head_ntype_embeddings = head_ntype_embeddings[:batch_sizes[self.head_node_type]]

            logits = self.classifier.forward(head_ntype_embeddings, h_dict=h_out)
        else:
            logits = h_out[self.head_node_type]

        if return_embeddings:
            return h_out, logits
        else:
            return logits

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.update_node_clf_metrics(self.train_metrics, y_pred, y_true, weights)

        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=2)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss, )

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=2)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel, classes=self.dataset.classes)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)

        self.log("test_loss", test_loss)

        return test_loss




class MLP(AFPNodeClf):
    def __init__(self, hparams, dataset, metrics: Union[List[str], Dict[str, List[str]]], *args, **kwargs):
        super().__init__(hparams, dataset, metrics, *args, **kwargs)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel

        self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)
        self.classifier = DenseClassification(hparams)
        self.criterion = ClassificationLoss(
            loss_type=hparams.loss_type,
            n_classes=dataset.n_classes,
            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                 'use_class_weights' in hparams and hparams.use_class_weights else None,
            pos_weight=dataset.pos_weight if hasattr(dataset, "pos_weight") and
                                             'use_pos_weights' in hparams and hparams.use_pos_weights else None,
            multilabel=dataset.multilabel,
            reduction=hparams.reduction if "reduction" in hparams else "mean")

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[Union[str, Tuple[str]], Union[Tensor, int]]]],
                return_embeddings=False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        h_out = {}
        if len(h_out) < len(inputs["global_node_index"].keys()):
            sp_tensor = inputs["x_dict"][self.head_node_type]
            inputs["x_dict"][self.head_node_type].storage._value = torch.ones_like(sp_tensor.storage._value,
                                                                                   device=self.device)
            h_out = self.encoder.forward(inputs["x_dict"], global_node_index=inputs["global_node_index"])

        if hasattr(self, "classifier"):
            head_ntype_embeddings = h_out[self.head_node_type]
            if "batch_size" in inputs and self.head_node_type in inputs["batch_size"]:
                head_ntype_embeddings = head_ntype_embeddings[:inputs["batch_size"][self.head_node_type]]

            logits = self.classifier.forward(head_ntype_embeddings, h_dict=h_out)
        else:
            logits = h_out[self.head_node_type]

        if return_embeddings:
            return h_out, logits
        else:
            return logits

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        scores = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], scores, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        loss = self.criterion.forward(scores, y_true, weights=weights)
        self.update_node_clf_metrics(self.train_metrics, scores, y_true, weights)

        if batch_nb % 100 == 0 and isinstance(self.train_metrics, Metrics):
            logs = self.train_metrics.compute_metrics()
        else:
            logs = {}

        self.log("loss", loss, on_step=True)
        self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        scores = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], scores, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        scores = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], scores, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)
        self.log("test_loss", test_loss)

        return test_loss

    @torch.no_grad()
    def predict(self, dataloader: DataLoader, node_names: pd.Index = None, save_betas=False, **kwargs) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """

        Args:
            dataloader ():
            node_names (): an pd.Index or np.ndarray to map node indices to node names for `head_node_type`.
            **kwargs ():

        Returns:
            targets, scores, embeddings, ntype_nids
        """
        if isinstance(self, RelationAttention):
            self.reset()

        y_trues = []
        y_preds = []
        embs = []
        nids = []

        for batch in tqdm.tqdm(dataloader, desc='Predict dataloader'):
            X, y_true, weights = to_device(batch, device=self.device)
            h_dict, logits = self.forward(X, return_embeddings=True)
            y_pred = activation(logits, loss_type=self.hparams['loss_type'])

            y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
            y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
            idx = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights, return_index=True)

            y_true = y_true[idx]
            y_pred = y_pred[idx]
            emb: Tensor = h_dict[self.head_node_type][idx]

            global_node_index = X["global_node_index"][-1] \
                if isinstance(X["global_node_index"], (list, tuple)) else X["global_node_index"]
            nid = global_node_index[self.head_node_type][idx]

            # Convert all to CPU device
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            embs.append(emb.cpu().numpy())
            nids.append(nid.cpu().numpy())

        targets = np.concatenate(y_trues, axis=0)
        scores = np.concatenate(y_preds, axis=0)
        embeddings = np.concatenate(embs, axis=0)
        node_ids = np.concatenate(nids, axis=0)

        if node_names is not None:
            index = node_names[node_ids]
        else:
            index = pd.Index(node_ids, name="nid")

        targets = pd.DataFrame(targets, index=index, columns=self.dataset.classes)
        scores = pd.DataFrame(scores, index=index, columns=self.dataset.classes)
        embeddings = pd.DataFrame(embeddings, index=index)
        ntype_nids = {self.head_node_type: node_ids}

        return targets, scores, embeddings, ntype_nids


class HGTNodeClf(AFPNodeClf):
    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        super(AFPNodeClf, self).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"HGT-{hparams.n_layers}"
        self.collate_fn = collate_fn
        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(dataset.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = HGT(embedding_dim=hparams.embedding_dim, num_layers=hparams.n_layers,
                            num_heads=hparams.attn_heads,
                            node_types=dataset.G.node_types, metadata=dataset.G.metadata())

        # Output layer
        if "cls_graph" in hparams and hparams.cls_graph is not None:
            self.classifier = LabelGraphNodeClassifier(dataset, hparams)

        elif hparams.nb_cls_dense_size >= 0:
            self.classifier = DenseClassification(hparams)
        else:
            assert hparams.layer_pooling != "concat", "Layer pooling cannot be concat when output of network is a GNN"

        use_cls_weight = 'use_class_weights' in hparams and hparams.use_class_weights
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight \
                                                if use_cls_weight and hasattr(dataset, "class_weight") else None,
                                            multilabel=dataset.multilabel,
                                            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr


class RGCNNodeClf(AFPNodeClf):
    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        super(AFPNodeClf, self).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"RGCN-{hparams.n_layers}"
        self.collate_fn = collate_fn
        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(dataset.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.relations = tuple(m for m in dataset.metapaths if m[0] == m[-1] and m[0] in self.head_node_type)
        print(self.name(), self.relations)
        self.embedder = RGCN(hparams.embedding_dim, num_layers=hparams.n_layers, num_relations=len(self.relations))

        # Output layer
        self.classifier = DenseClassification(hparams)

        use_cls_weight = 'use_class_weights' in hparams and hparams.use_class_weights
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight \
                                                if use_cls_weight and hasattr(dataset, "class_weight") else None,
                                            multilabel=dataset.multilabel,
                                            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[Union[str, Tuple[str]], Union[Tensor, int]]]],
                return_embeddings=False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        batch_sizes = inputs["batch_size"]
        if isinstance(self.classifier, LabelNodeClassifer):
            # Ensure we count betas and alpha values for the `pred_ntypes`
            batch_sizes = batch_sizes | self.classifier.class_sizes

        h_out = {}
        #  Node feature or embedding input
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  split_size=math.sqrt(self.hparams.batch_size // 4)))
        # Node sequence features
        global_node_index = {ntype: nids for ntype, nids in inputs["global_node_index"].items() \
                             if ntype == self.head_node_type}
        if len(h_out) < len(global_node_index.keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_index=global_node_index)
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        # Build multi-relational edge_index
        edge_index_dict = {m: get_edge_index_values(eid)[0] for m, eid in inputs["edge_index_dict"].items() \
                           if m in self.relations}
        edge_value_dict = {m: get_edge_index_values(eid)[1] for m, eid in inputs["edge_index_dict"].items() \
                           if m in self.relations}
        edge_index = torch.cat([edge_index_dict[m] for m in self.relations], dim=1)
        edge_value = torch.cat([edge_value_dict[m] for m in self.relations], dim=0)
        edge_type = torch.tensor([self.relations.index(m) for m in self.relations \
                                  for i in range(edge_index_dict[m].size(1))], dtype=torch.long,
                                 device=edge_index.device)

        edge_index = SparseTensor.from_edge_index(edge_index, edge_value,
                                                  sparse_sizes=(global_node_index[self.head_node_type].size(0),
                                                                global_node_index[self.head_node_type].size(0)),
                                                  trust_data=True)

        h_out[self.head_node_type] = self.embedder.forward(h_out[self.head_node_type], edge_index=edge_index,
                                                           edge_type=edge_type)

        # Node classification
        if hasattr(self, "classifier"):
            head_ntype_embeddings = h_out[self.head_node_type]
            if "batch_size" in inputs and self.head_node_type in batch_sizes:
                head_ntype_embeddings = head_ntype_embeddings[:batch_sizes[self.head_node_type]]

            logits = self.classifier.forward(head_ntype_embeddings, h_dict=h_out)
        else:
            logits = h_out[self.head_node_type]

        if return_embeddings:
            return h_out, logits
        else:
            return logits


