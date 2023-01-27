import glob
import logging
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Union

import pandas as pd
from ruamel import yaml

from moge.network.hetero import HeteroNetwork
from experiments.datasets.CAFA import build_cafa_dataset


def load_node_dataset(name: str, method, hparams: Namespace,
                      dataset_path: str = None,
                      latte2go_yaml='experiments/configs/latte2go.yaml'):

    if method == 'DeepGraphGO':
        dataset = None  # will load in method

    elif "HUMAN_MOUSE" in name or "MULTISPECIES" in name:
        extra_args = {}
        if name == 'HUMAN_MOUSE':
            dataset_path = 'data/heteronetwork/DGG_HUMAN_MOUSE_MirTarBase_TarBase_LncBase_RNAInter_STRINGphyssplit_BioGRID_mRNAprotein_transcriptlevel.HeteroNetwork'
        elif name == "MULTISPECIES":
            dataset_path = 'data/heteronetwork/DGG_MirTarBase_TarBase_LncBase_RNAInter_STRINGphyssplit_BioGRID_mRNAprotein_transcriptlevel.HeteroNetwork'

        mlb_path = os.path.expanduser('~/Bioinformatics_ExternalData/LATTE2GO')
        mlb_paths = glob.glob(f'{mlb_path}/{hparams.dataset}-{hparams.pred_ntypes}/go_id.mlb')
        if mlb_paths:
            hparams.mlb_path = mlb_paths[0]

        with open(latte2go_yaml, 'r') as f:
            hparams.__dict__.update(yaml.safe_load(f))

        if 'LATTE2GO' in hparams.method and hparams.pred_ntypes not in hparams.ntype_subset:
            hparams.ntype_subset = hparams.ntype_subset + ' ' + hparams.pred_ntypes

        dataset = build_cafa_dataset('UniProt', dataset_path=dataset_path, hparams=hparams, **extra_args)

    else:
        raise Exception(f"dataset {name} not found")

    return dataset


