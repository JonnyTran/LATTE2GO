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
from run.datasets.CAFA import build_cafa_dataset


def load_node_dataset(name: str, method, hparams: Namespace,
                      dataset_path: Path = "dataset"):

    if method == 'DeepGraphGO':
        dataset = None  # will load in method

    elif "HUMAN_MOUSE" in name or "MULTISPECIES" in name:
        extra_args = {}
        if name == 'HUMAN_MOUSE':
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_HUMAN_MOUSE_MirTarBase_TarBase_LncBase_RNAInter_STRINGphyssplit_BioGRID_mRNAprotein_transcriptlevel.HeteroNetwork'
        elif name == 'HUMAN_MOUSE_unsplit':
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_HUMAN_MOUSE_MirTarBase_TarBase_LncBase_RNAInter_STRING_BioGRID_mRNAprotein_transcriptlevel.network.pickle'
            extra_args['save'] = False
            extra_args['rebuild'] = True
        elif name == "MULTISPECIES":
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_MirTarBase_TarBase_LncBase_RNAInter_STRINGphyssplit_BioGRID_mRNAprotein_transcriptlevel.HeteroNetwork'
        elif name == "MULTISPECIES_unsplit":
            dataset_path = '~/PycharmProjects/Multiplex-Graph-Embedding/data/heteronetwork/DGG_MirTarBase_TarBase_LncBase_RNAInter_STRING_BioGRID_mRNAprotein_transcriptlevel.network.pickle'
            extra_args['save'] = False
            extra_args['rebuild'] = True

        mlb_path = os.path.expanduser('~/Bioinformatics_ExternalData/LATTE2GO')
        mlb_paths = glob.glob(f'{mlb_path}/{hparams.dataset}-{hparams.pred_ntypes}/go_id.mlb')
        if mlb_paths:
            hparams.mlb_path = mlb_paths[0]

        with open('run/configs/_latte2go_helper.yaml', 'r') as f:
            hparams.__dict__.update(yaml.safe_load(f))

        if 'LATTE2GO' in hparams.method and hparams.pred_ntypes not in hparams.ntype_subset:
            hparams.ntype_subset = hparams.ntype_subset + ' ' + hparams.pred_ntypes

        dataset = build_cafa_dataset('UniProt', dataset_path=dataset_path, hparams=hparams, **extra_args)


    elif 'UNIPROT' in name.upper() and (
            isinstance(dataset_path, HeteroNetwork) or (isinstance(dataset_path, str) and ".pickle" in dataset_path)):
        dataset = build_cafa_dataset(name, dataset_path=dataset_path, hparams=hparams)

    else:
        raise Exception(f"dataset {name} not found")

    return dataset


