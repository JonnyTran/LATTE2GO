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
        if name == 'HUMAN_MOUSE':
            dataset_path = 'data/UniProt.InterPro.HUMAN_MOUSE.DGG.parents/'
        elif name == "MULTISPECIES":
            dataset_path = 'data/UniProt.InterPro.MULTISPECIES.DGG.parents/'
        else:
            raise Exception(f"Dataset name {name} not supported.")

        if hparams.pred_ntypes == "biological_process":
            hparams.mlb_path = f'data/DeepGraphGO/data/bp_go.mlb'
        elif hparams.pred_ntypes == "molecular_function":
            hparams.mlb_path = f'data/DeepGraphGO/data/mf_go.mlb'
        elif hparams.pred_ntypes == "cellular_component":
            hparams.mlb_path = f'data/DeepGraphGO/data/cc_go.mlb'
        else:
            # Prediction on classes of two or more ontologies
            hparams.mlb_path = f'data/all_go.mlb'

        with open(latte2go_yaml, 'r') as f:
            hparams.__dict__.update(yaml.safe_load(f))

        if 'LATTE2GO' in hparams.method and hparams.pred_ntypes not in hparams.ntype_subset:
            hparams.ntype_subset = hparams.ntype_subset + ' ' + hparams.pred_ntypes

        dataset = build_cafa_dataset('UniProt', dataset_path=dataset_path, hparams=hparams)

    else:
        raise Exception(f"dataset {name} not found")

    return dataset


