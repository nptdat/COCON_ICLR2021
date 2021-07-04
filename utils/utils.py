from collections import OrderedDict
import logging
import random
import numpy as np
import shutil
import glob
import os
import re
from typing import List

import torch

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/22
def to_one_hot(y, n_dims=None, debug=False):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).reshape(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    if debug:
        y_compare = torch.argmax(y_one_hot, dim=-1)
        logger.info( "y_compare: {}".format(y_compare))
        logger.info( "u: {}".format(y))

    return y_one_hot


def _clear_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)

    for checkpoint in checkpoints_sorted:
        logger.info("Deleting older checkpoint [{}] before rerunning training".format(checkpoint))
        shutil.rmtree(checkpoint)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def fix_state_dict_naming(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'con2' in key:
            new_key = key.replace('con2', 'cocon')
        # new_key = key_transformation(key)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict