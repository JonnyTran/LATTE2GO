import datetime
import os
import random
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from logzero import logger
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping

sys.path.insert(0, "../LATTE2GO/") # Needed to import moge

from moge.model.PyG.node_clf import LATTEFlatNodeClf, HGTNodeClf, MLP, RGCNNodeClf
from moge.model.tensor import tensor_sizes

from experiments.datasets.deepgraphgo import build_deepgraphgo_model
from experiments.utils import parse_yaml_config, select_empty_gpus
from experiments.datasets.load import load_node_dataset


def train(hparams):
    pytorch_lightning.seed_everything(hparams.seed)

    NUM_GPUS = hparams.num_gpus
    USE_AMP = True
    MAX_EPOCHS = 1000
    MIN_EPOCHS = getattr(hparams, 'min_epochs', 60)

    ### Dataset
    dataset = load_node_dataset(hparams.dataset, hparams.method, hparams=hparams)
    if dataset is not None:
        hparams.n_classes = dataset.n_classes
        hparams.head_node_type = dataset.head_node_type
        print("dataset.pred_ntypes", dataset.pred_ntypes)
        print(tensor_sizes(class_indices=dataset.class_indices))


    ### Callbacks
    callbacks = []
    if hparams.dataset.upper() in ['UNIPROT', "MULTISPECIES", "HUMAN_MOUSE"] or \
            "HUMAN_MOUSE" in hparams.dataset or "MULTISPECIES" in hparams.dataset:
        METRICS = ["BPO_aupr", "BPO_fmax", "CCO_aupr",
                   "CCO_fmax", "MFO_aupr", "MFO_fmax",
                   "BPO_smin", "CCO_smin", "MFO_smin"]
        early_stopping_args = dict(monitor='val_aupr', mode='max', patience=hparams.early_stopping)
    else:
        METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]
        early_stopping_args = dict(monitor='val_loss', mode='min')


    if hparams.method == "HGT":
        USE_AMP = False
        default_args = {
            "embedding_dim": 512,
            "n_layers": 2,
            "batch_size": 2 ** 11,
            "activation": "relu",
            "attn_heads": 4,
            "attn_dropout": 0.2,
            "dropout": 0.5,
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_norm": True,
            "use_class_weights": False,
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-2,
        }

        model = HGTNodeClf(Namespace(**default_args), dataset, metrics=METRICS)

    elif hparams.method == "RGCN":
        USE_AMP = False
        default_args = {
            "embedding_dim": 512,
            "n_layers": 2,
            "batch_size": 2 ** 11,
            "activation": "relu",
            "dropout": 0.5,
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "lr": 1e-3,
        }

        model = RGCNNodeClf(Namespace(**default_args), dataset, metrics=METRICS)


    elif "LATTE" in hparams.method:
        USE_AMP = False
        early_stopping_args['monitor'] = 'val_aupr_mean'
        early_stopping_args['patience'] = 35

        if hparams.method.endswith("-1"):
            t_order = 1
            batch_order = 11

        elif hparams.method.endswith("-2"):
            t_order = 2
            batch_order = 10

        elif hparams.method.endswith("-3"):
            t_order = 3
            batch_order = 10
        else:
            raise Exception()

        batch_size = int(2 ** batch_order)
        n_layers = 2
        dataset.neighbor_sizes = [2048, ] * n_layers

        if t_order > 1:
            ntype_metapaths = pd.DataFrame(dataset.metapaths, columns=['src', 'etype', 'dst']).query(
                f'src == dst and src in {dataset.pred_ntypes}')
            ntype_metapaths = ntype_metapaths.groupby('src')['etype'].unique().to_dict()
            filter_metapaths = {ntype: [tuple([etype, ] * hops) for etype in etypes for hops in range(2, t_order + 1)] \
                                for ntype, etypes in ntype_metapaths.items()}
        else:
            filter_metapaths = None

        default_args = {
            "embedding_dim": 512,
            "layer_pooling": "concat",

            "n_layers": n_layers,
            "t_order": t_order,
            'neighbor_sizes': dataset.neighbor_sizes,
            "batch_size": batch_size,

            "filter_metapaths": filter_metapaths,
            "filter_self_metapaths": True,

            "attn_heads": 8,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.0,

            "batchnorm": False,
            "layernorm": True,
            "activation": "relu",
            "dropout": 0.0,

            "head_node_type": dataset.head_node_type,

            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "lr_annealing": None,
        }
        hparams.__dict__.update(default_args)
        model = LATTEFlatNodeClf(hparams, dataset, metrics=METRICS)

    elif 'DeepGraphGO' == hparams.method:
        USE_AMP = False
        model = build_deepgraphgo_model(hparams, base_path='data/DeepGraphGO')

    elif 'MLP' == hparams.method:
        dataset.neighbor_sizes = [0]
        hparams.__dict__.update({
            "embedding_dim": 512,
            "n_layers": len(dataset.neighbor_sizes),
            'neighbor_sizes': dataset.neighbor_sizes,
            "batch_size": 2 ** 12,
            "dropout": 0.0,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "lr": 1e-3,
        })

        model = MLP(hparams, dataset=dataset, metrics=METRICS)

    else:
        raise Exception(f"Unknown model {hparams.model}")

    model.train_metrics.metrics = {}

    tags = [] + hparams.dataset.split(" ")
    if hasattr(hparams, "namespaces"):
        tags.extend(hparams.namespaces)
    if hasattr(dataset, 'tags'):
        tags.extend(dataset.tags)

    logger = WandbLogger(name=getattr(hparams, 'method', model.name()),
                         tags=list(set(tags)),
                         anonymous=True)
    logger.log_hyperparams(tensor_sizes(hparams))

    if hparams.early_stopping:
        callbacks.append(EarlyStopping(strict=False, **early_stopping_args))

    if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int):
        GPUS = [hparams.gpu]
    elif NUM_GPUS:
        GPUS = select_empty_gpus(NUM_GPUS)

    trainer = Trainer(
        accelerator='cuda',
        devices=GPUS,
        # enable_progress_bar=False,
        # auto_scale_batch_size=True if hparams.n_layers > 2 else False,
        max_epochs=MAX_EPOCHS,
        min_epochs=MIN_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        max_time=datetime.timedelta(hours=hparams.hours) \
            if hasattr(hparams, "hours") and isinstance(hparams.hours, (int, float)) else None,
        # plugins='deepspeed' if NUM_GPUS > 1 else None,
        # accelerator='ddp_spawn',
        # plugins='ddp_sharded'
        precision=16 if USE_AMP else 32,
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

def update_hparams_from_env(hparams: Namespace, dataset=None):
    updates = {}
    if 'batch_size'.upper() in os.environ:
        updates['batch_size'] = int(os.environ['batch_size'.upper()])
        if hasattr(dataset, 'neighbor_sizes'):
            dataset.neighbor_sizes = [int(n * (updates['batch_size'] / hparams['batch_size'])) \
                                      for n in dataset.neighbor_sizes]

    if 'n_neighbors'.upper() in os.environ:
        updates['n_neighbors'] = int(os.environ['n_neighbors'.upper()])

    logger.info(f"Hparams updates from ENV: {updates}")

    if isinstance(hparams, Namespace):
        hparams.__dict__.update(updates)
    elif isinstance(hparams, dict):
        hparams.update(updates)
    return hparams


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--method', type=str, default="LATTE2GO-2")

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--inductive', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default="MULTISPECIES")
    parser.add_argument('--pred_ntypes', type=str, default="biological_process")
    parser.add_argument('--ntype_subset', type=str, default="Protein")

    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=15)

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=random.randint(0, int(1e4)))
    parser.add_argument('--hours', type=int, default=23)

    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False,
                        default='experiments/configs/latte2go.yaml')
    # add all the available options to the trainer
    args = parse_yaml_config(parser)
    train(args)
