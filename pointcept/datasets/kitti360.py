"""
KITTI-360 dataset

Author: Zijian Yu
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy

from .builder import DATASETS
from .defaults import DefaultDataset

from pointcept.utils.cache import shared_dict


@DATASETS.register_module()
class Kitti360Dataset(DefaultDataset):
    def get_data_list(self):
        data_list = glob.glob(os.path.join(self.data_root, self.split, "*", "*"))
        return data_list

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        window_id = os.path.basename(data_path)
        drive_id = os.path.basename(os.path.dirname(data_path))
        return f"{drive_id}-{window_id}"
