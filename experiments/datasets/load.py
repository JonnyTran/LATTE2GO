from argparse import Namespace

from ruamel import yaml

from experiments.datasets.CAFA import build_cafa_dataset


def load_node_dataset(name: str, method:str, hparams: Namespace,
                      latte2go_yaml='experiments/configs/latte2go.yaml',
                      save_path='data/'):

    if method == 'DeepGraphGO':
        dataset = None  # will load in method

    elif "HUMAN_MOUSE" in name or "MULTISPECIES" in name:
        if name == 'HUMAN_MOUSE':
            dataset_path = 'data/UniProt.InterPro.HUMAN_MOUSE.DGG.HeteroNetwork/'
        elif name == "MULTISPECIES":
            dataset_path = 'data/UniProt.InterPro.MULTISPECIES.DGG.HeteroNetwork/'
        else:
            raise Exception(f"Dataset name {name} not supported.")

        # Update additional params from `experiments/configs/latte2go.yaml`
        with open(latte2go_yaml, 'r') as f:
            hparams.__dict__.update(yaml.safe_load(f))

        # If using LATTE2GO, include the GO ntypes to the hetero graph ntypes
        if 'LATTE2GO' in hparams.method and hparams.pred_ntypes not in hparams.ntype_subset:
            hparams.ntype_subset = hparams.ntype_subset + ' ' + hparams.pred_ntypes

        dataset = build_cafa_dataset('UniProt', dataset_path=dataset_path, hparams=hparams, save_path=save_path)

    else:
        raise Exception(f"dataset {name} not found")

    return dataset


