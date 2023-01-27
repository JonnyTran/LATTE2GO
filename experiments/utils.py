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

        opt = {k: v for k, v in opt.items() if k not in args}
        args_dict.update(opt)
        args = Namespace(**args_dict)

        print("Configs:")
        pprint(opt)
        print()

    return args



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


