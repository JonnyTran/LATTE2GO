import os
from argparse import ArgumentParser, Namespace
from pprint import pprint
from typing import Union, List

import dgl
import dill
import pynvml
import torch
import yaml
from logzero import logger


def parse_yaml_config(parser: ArgumentParser) -> Namespace:
    """

    Args:
        parser ():

    Returns:

    """
    args = parser.parse_args()
    # yaml priority is higher than args
    if isinstance(getattr(args, 'config', None), str) and os.path.exists(args.config):
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args_dict = args.__dict__

        opt = {k: v for k, v in opt.items if k not in args}
        args_dict.update(opt)
        args = Namespace(**args_dict)

        print("Configs:")
        pprint(opt)
        print()

    return args


def adjust_batch_size(hparams):
    batch_size = hparams.batch_size
    if batch_size < 0: return batch_size

    if hparams.n_neighbors > 256:
        batch_size = batch_size // (hparams.n_neighbors // 128)
    if hparams.embedding_dim > 128:
        batch_size = batch_size // (hparams.embedding_dim // 128)
    if hparams.n_layers > 2:
        batch_size = batch_size // (hparams.n_layers - 1)

    logger.info(f"Adjusted batch_size to", batch_size)

    return int(batch_size)


def select_empty_gpus(num_gpus=1) -> List[int]:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    avail_device = []
    for i in range(deviceCount):
        device = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(device)

        avail_device.append((info.free / info.total, i))

    best_gpu = max(avail_device)[1]
    return [best_gpu]


