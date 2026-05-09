"""
Instance Segmentation / Interactive Segmentation dataset augmentation

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import random
import torch
import numpy as np
from collections.abc import Mapping
from ..transform import TRANSFORMS, index_operator, GridSample


"""
省得改写transform了, 为了满足预采样的需要
- 直接复用 sampled_index 来避免丢弃采样点
    - RandomDropout、GridSample(train)
- 将idx转为mask, 来适应需要做 index_operator 的transform
    - index_operator 是需要大幅度换顺序的
        - 要求处理的数据, 在 dim0 上位置一致, 直接 indexing
    - 增加 index_valid_keys, 让 index_operator 直接处理
    - SphereCrop、ShufflePoint、CropBoundary
- 其它 transform 只需要改变值, 不会改变点排序, 所以不需要转换
- 保留 mask, 方便 concat mini batch, 不需要 reindex
"""


@TRANSFORMS.register_module()
class SampledIndex2Mask(object):
    def __call__(self, data_dict):
        if "sampled_index" in data_dict:
            mask = np.zeros_like(data_dict["segment"]).astype(bool)
            mask[data_dict["sampled_index"]] = True
            data_dict["sampled_mask"] = mask
            data_dict["index_valid_keys"].append("sampled_mask")
        return data_dict


@TRANSFORMS.register_module()
class InstanceCrop(object):
    """随机将一部分 instance 整体 crop 掉, 改变 instance 的出现频率"""

    def __init__(
        self,
        sample_rate=None,
        mode="random",
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
    ):
        self.sample_rate = sample_rate
        assert mode in ["random", "background"]
        self.mode = mode
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background

    def __call__(self, data_dict):
        # TODO
        pass


@TRANSFORMS.register_module()
class GridSampleISeg(GridSample):
    """更改了 sampled_index 的处理逻辑, 保证 hash key 唯一"""

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                # 为每个sampled_index按照key找到对应的idx_unique位置并替换
                sampled_keys = key[data_dict["sampled_index"]]
                unique_keys = key[idx_unique]
                # 键为unique_key，值为idx_unique中的位置
                unique_key_pos = {k: i for i, k in enumerate(unique_keys)}
                for i, idx in enumerate(data_dict["sampled_index"]):
                    sampled_key = sampled_keys[i]
                    if sampled_key in unique_key_pos:  # 应该总是存在
                        pos = unique_key_pos[sampled_key]
                        idx_unique[pos] = idx
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            raise NotImplementedError
        else:
            raise NotImplementedError
