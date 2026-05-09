"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import cv2
import glob
import copy
import random
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Mapping, Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from .utils import collate_fn


# TODO 没写完, 需要在scannet中使img的Tensor.shape保持一致, 能在dim0上cat
def fusion_collate_fn(batch, mix_prob=0):
    """for fusion dataset"""
    img_dict_list = [b[0] for b in batch]
    pcd_dict_list = [b[1] for b in batch]
    assert isinstance(
        img_dict_list[0], Mapping
    ), "currently, only support input_dict, rather than input_list"
    assert isinstance(
        pcd_dict_list[0], Mapping
    ), "currently, only support input_dict, rather than input_list"
    img_dict = collate_fn(img_dict_list)
    pcd_dict = collate_fn(pcd_dict_list)
    if "offset" in pcd_dict.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            pcd_dict["offset"] = torch.cat(
                [pcd_dict["offset"][1:-1:2], pcd_dict["offset"][-1].unsqueeze(0)], dim=0
            )
    return img_dict, pcd_dict


def num_to_natural(group_ids):
    """
    Change the group number to natural number arrangement

    去掉中间没有出现的id，使得id连续
    """
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


@DATASETS.register_module()
class ScanNetImageDataset(Dataset):
    """
    魔改，RGBD，200和20写到一起了
    - task_type：“semantic_20”、“semantic_200”、“instance”、“semantic_nyu40”
    """

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        task_type="semantic_nyu40",
    ):
        super(ScanNetImageDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.task_type = task_type

        self.scene_list = self.get_scene_list()
        self.img_name_list = self.get_image_name_list()
        # 内参
        self.intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics.txt"))
        self.color_image_list, self.img_num = self.get_image_list(
            split="color", suffix=".jpg"
        )
        self.depth_image_list, _ = self.get_image_list(split="depth")
        # 按照类别读label，后面重写数据预处理的时候再整合 TODO
        if task_type == "semantic_nyu40":
            self.label_image_list, _ = self.get_image_list(split="label")
        else:
            self.label_image_list, _ = self.get_image_list(split=task_type)
        # id映射，后面修改统一 TODO
        if task_type == "semantic_20":
            self.class2id = np.array(VALID_CLASS_IDS_20)
        else:
            self.class2id = np.array(VALID_CLASS_IDS_200)

        self.pose_list = self.get_pose_list()

        self.image_shape = cv2.imread(
            self.color_image_list[os.path.split(self.scene_list[0])[-1]][0],
            cv2.IMREAD_UNCHANGED,
        ).shape[:2]

        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} scene, {} images x {} samples in {} set.".format(
                len(self.scene_list), self.img_num.sum(), self.loop, split
            )
        )

    def get_scene_list(self):
        """获取scene的path列表"""
        if isinstance(self.split, str):
            scene_list = glob.glob(os.path.join(self.data_root, self.split, "scene*"))
        elif isinstance(self.split, Sequence):
            scene_list = []
            for split in self.split:
                scene_list += glob.glob(os.path.join(self.data_root, split, "scene*"))
        else:
            raise NotImplementedError
        return scene_list

    def get_image_name_list(self, split="color", suffix=".jpg"):
        if isinstance(self.scene_list, list):
            img_name_list = {}
            for scene in self.scene_list:
                scene_name = os.path.split(scene)[-1]
                img_path_list = glob.glob(os.path.join(scene, split, "*" + suffix))
                img_name_list[scene_name] = [
                    os.path.split(x)[-1].split(".")[0] for x in img_path_list
                ]
                img_name_list[
                    scene_name
                ].sort()  # 有必要排序，因为需要color、depth、label对应起来
        else:
            raise NotImplementedError
        return img_name_list

    def get_image_list(self, split="color", suffix=".png"):
        """获取image的path列表，split为img所在文件夹名，返回image_list[scene_name][image_index], img_num[scene_index]"""
        if isinstance(self.scene_list, list):
            image_list = {}
            img_num = []
            for scene in self.scene_list:
                scene_name = os.path.split(scene)[-1]
                image_list[scene_name] = glob.glob(
                    os.path.join(scene, split, "*" + suffix)
                )
                image_list[
                    scene_name
                ].sort()  # 有必要排序，因为需要color、depth、label对应起来
                img_num.append(len(image_list[scene_name]))
        else:
            raise NotImplementedError
        return image_list, np.asarray(img_num, dtype=int)

    def get_pose_list(self):
        """返回pose_list[scene_name][pose_index]"""
        if isinstance(self.scene_list, list):
            pose_list = {}
            for scene in self.scene_list:
                scene_name = os.path.split(scene)[-1]
                pose_list[scene_name] = glob.glob(os.path.join(scene, "pose", "*.txt"))
                pose_list[scene_name].sort()
        else:
            raise NotImplementedError
        return pose_list

    def get_data_by_scene_name(self, scene_name):
        """
        按照scene_name获取一个scene的image & pose
        """
        img_path_list = self.color_image_list[scene_name]
        depth_path_list = self.depth_image_list[scene_name]
        label_path_list = self.label_image_list[scene_name]
        pose_path_list = self.pose_list[scene_name]
        img_name_list = self.img_name_list[scene_name]

        # TODO 加cache
        if not self.cache:
            img_list = []
            depth_list = []
            label_list = []
            pose_list = []
            for img_path, depth_path, label_path, pose_path in zip(
                img_path_list, depth_path_list, label_path_list, pose_path_list
            ):
                # 注意cv2读的是BGR的, 需要转换
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                depth = cv2.imread(depth_path, -1)
                label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                pose = np.loadtxt(pose_path)
                img_list.append(img)
                depth_list.append(depth)
                label_list.append(label)
                pose_list.append(pose)

        data_dict = dict(
            img_list=img_list,
            depth_list=depth_list,
            label_list=label_list,
            pose_list=pose_list,
            img_name_list=img_name_list,
            scene_id=scene_name,
        )
        return data_dict

    def get_data_by_class_tags(self, scene_name, class_tag_list):
        raise NotImplementedError

    def get_data_by_scene(self, idx):
        """
        按照idx获取一个scene的image & pose
        """
        scene_path = self.scene_list[idx % len(self.scene_list)]
        scene_name = os.path.split(scene_path)[-1]
        return self.get_data_by_scene_name(scene_name)

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data_by_scene(idx)
        data_dict = self.transform(
            data_dict
        )  # build后, 如果transform是None, 就会返回原数据
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data_by_scene(idx)
        # segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        # TODO: augmentation
        # data_dict_list = []
        # for aug in self.aug_transform:
        #     data_dict_list.append(
        #         aug(deepcopy(data_dict))
        #     )

        # input_dict_list = []
        # for data in data_dict_list:
        #     data_part_list = self.test_voxelize(data)
        #     for data_part in data_part_list:
        #         if self.test_crop:
        #             data_part = self.test_crop(data_part)
        #         else:
        #             data_part = [data_part]
        #         input_dict_list += data_part

        # for i in range(len(input_dict_list)):
        #     input_dict_list[i] = self.post_transform(input_dict_list[i])
        # data_dict = dict(fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx))
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


# TODO 按照scene id、class id, 随机取一定数目的包含完整mask的图片


@DATASETS.register_module()
class ScanNetFusionDataset(Dataset):
    """
    魔改，RGBD+pointcloud
    - 确保collect的属性里有"color"和"scene_id"
    - 不要augmentation，因为需要联系image和点云
    """

    def __init__(self, image_cfg=None, pointcloud_cfg=None, loop=1):
        super(ScanNetFusionDataset, self).__init__()
        image_cfg.loop = loop
        pointcloud_cfg.loop = loop
        self.image_dataset = DATASETS.build(image_cfg)
        self.pointcloud_dataset = DATASETS.build(pointcloud_cfg)
        assert self.image_dataset.test_mode == self.pointcloud_dataset.test_mode
        self.test_mode = self.image_dataset.test_mode
        self.class2id = self.pointcloud_dataset.class2id

    @staticmethod
    def proj_pcd_to_img(pcd_dict, intrinsic, color_img, depth_img, pose):
        """
        投影scene点云，到color_img对应的图像上
        - pcd_dict: data dict, 包含场景的点云, 需要属性里有'coord'、"color"
        - intrinsic: 内参, np array shape as (4,4)
        - color_img, depth_img: 需要大小一致
        - pose: cam to pointcloud 的3D transform矩阵, np array shape as (4,4)
        - 返回将点云和图像关联起来的信息:
            - sampled_points[idx]: 对应的关联的图像点坐标+深度
            - sampled_colors[idx]: 对应的关联的点云颜色
            - sampled_index[idx]: 关联的点云点在原点云数据的index
            - scene: 原点云data dict
            - image_projected: 投影后的图片
        """
        if pose[0, 0] == float("-inf"):
            # 如果没有位姿, 返回 None
            return None
        image_shape = color_img.shape[:2]
        points = pcd_dict["coord"]
        colors = pcd_dict["color"]
        sampled_index = np.arange(points.shape[0])  # 筛选后的点在原数据的index
        # 点云 -> cam frame -> img frame
        pcd = np.dot(
            np.linalg.inv(pose), np.hstack([points, np.ones((points.shape[0], 1))]).T
        )
        pcd = pcd / pcd[3]
        pcd = np.dot(intrinsic[:3], pcd)
        pcd[:2] = pcd[:2] / pcd[2]  # 归一化且保留depth
        # 裁剪出投影点在图像范围内的部分
        crop_points = pcd[
            :,
            (pcd[0] > 0)
            * (pcd[0] < image_shape[1])
            * (pcd[1] > 0)
            * (pcd[1] < image_shape[0])
            * (pcd[2] > 0),
        ]
        crop_colors = colors[
            (pcd[0] > 0)
            * (pcd[0] < image_shape[1])
            * (pcd[1] > 0)
            * (pcd[1] < image_shape[0])
            * (pcd[2] > 0)
        ]
        crop_index = sampled_index[
            (pcd[0] > 0)
            * (pcd[0] < image_shape[1])
            * (pcd[1] > 0)
            * (pcd[1] < image_shape[0])
            * (pcd[2] > 0)
        ]
        crop_points[:2] = np.floor(crop_points[:2])
        # print(crop_index.shape)
        crop_points = crop_points.T
        # 构造投影的img, 自行修改depth的匹配容忍程度
        image_projected = np.zeros_like(color_img)
        sampled_index_2 = []
        depth_shift = 1000.0
        # cnt = 0
        import math

        for i in range(crop_points.shape[0]):
            if math.isclose(
                depth_img[int(crop_points[i, 1]), int(crop_points[i, 0])] / depth_shift,
                crop_points[i, 2],
                rel_tol=0.02,
                abs_tol=0.02,
            ):
                image_projected[int(crop_points[i, 1]), int(crop_points[i, 0])] = (
                    crop_colors[i] + 1
                ) * 127.5  # 这里默认做过color归一化
                sampled_index_2.append(i)
                # cnt += 1
        # print(cnt)
        sampled_points = crop_points[sampled_index_2]  # 投影、裁剪后的匹配的点云
        sampled_colors = crop_colors[sampled_index_2]  # 颜色
        sampled_index = crop_index[sampled_index_2]  # 在原数据中的index

        data_dict = dict(
            sampled_points=sampled_points,
            sampled_colors=sampled_colors,
            sampled_index=sampled_index,
            image_projected=image_projected,
        )
        return data_dict

    def get_fusion(self, img_dict):
        """
        投影scene的点云, 到该场景的所有image
        - img_dict: data dict, 包含场景的img, 属性有 'img_list', 'depth_list', 'label_list', 'pose_list', 'img_name_list', 'scene_id'
        - 根据 'scene_id', 寻址点云
        - 返回将点云和图像关联起来的信息; 若图像没有pose, 则删去该图像, img_dict传指针也更新
        """
        intrinsic = self.image_dataset.intrinsic
        scene_name = img_dict["scene_id"]
        scene = self.pointcloud_dataset.get_data_by_name(scene_name)
        idx_to_rm = []
        fusion_dict = dict(
            sampled_points=[], sampled_colors=[], sampled_index=[], image_projected=[]
        )
        for i in range(len(img_dict["img_list"])):
            # 遍历
            res = self.proj_pcd_to_img(
                pcd_dict=scene,
                intrinsic=intrinsic,
                color_img=img_dict["img_list"][i],
                depth_img=img_dict["depth_list"][i],
                pose=img_dict["pose_list"][i],
            )
            if res:
                for key in fusion_dict:
                    fusion_dict[key].append(res[key])
            else:
                # 如果没有pose, 则删去该帧
                idx_to_rm.append(i)
        # 执行删除
        for key in img_dict:
            if key != "scene_id":
                for i in idx_to_rm:
                    img_dict[key].pop(i)
        return fusion_dict

    @staticmethod
    def reproj_img_to_pcd(intrinsic, color_img, depth_img, pose, group_ids=None):
        """
        将image反投影回3D, 构建新的点云(不同于数据集中重建的点云)
        - intrinsic: 内参, np array shape as (4,4)
        - color_img, depth_img: 需要大小一致
        - pose: cam to pointcloud 的3D transform矩阵, np array shape as (4,4)
        - group_ids(optional): 将预测的mask image输进来做对应, shape as depth_img.shape
        """
        if pose[0, 0] == float("-inf"):
            # 如果没有位姿, 返回 None
            return None
        mask = depth_img != 0
        color_img_flatten = np.reshape(color_img[mask], [-1, 3])  # 展平
        if group_ids is not None:
            group_ids = group_ids[mask]  # 做mask后，shape as (69061,)，已经展平
        colors = np.zeros_like(color_img_flatten)
        colors[:, 0] = color_img_flatten[:, 2]  # BGR转RGB
        colors[:, 1] = color_img_flatten[:, 1]
        colors[:, 2] = color_img_flatten[:, 0]
        depth_shift = 1000.0
        # 构建(pixel_num, 3)形状, 3为uv_depth三个值
        x, y = np.meshgrid(
            np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]),
            np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]),
        )
        uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
        uv_depth[:, :, 0] = x
        uv_depth[:, :, 1] = y
        uv_depth[:, :, 2] = depth_img / depth_shift
        uv_depth = np.reshape(uv_depth, [-1, 3])
        uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
        # 反投影
        # intrinsic_inv = np.linalg.inv(intrinsic)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        bx = intrinsic[0, 3]
        by = intrinsic[1, 3]
        n = uv_depth.shape[0]
        points = np.ones((n, 4))
        X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
        Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
        # 反投影到cam下的坐标
        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = uv_depth[:, 2]
        # 投影到world
        points_world = np.dot(points, np.transpose(pose))  # pose是cam 2 world的
        group_ids = num_to_natural(group_ids)
        save_dict = dict(coord=points_world[:, :3], color=colors, group=group_ids)
        return save_dict

    def __getitem__(self, idx):
        """
        按idx寻址pcd
        图片太占内存了, 所以为了开多batch, 不能一次性返回scene对应的所有图片, 应该按scene中的class/instance, 找对应的img_num_per_mask=10张图片
        """
        scene = self.pointcloud_dataset[idx]  # 场景的data_dict
        # 然后按名字找对应的img, 如果直接__getitem__, 会是无序的, 对应不起来
        # if self.pointcloud_dataset.test_mode:
        #     scene_name = scene['fragment_list'][0]["scene_id"]
        # else:
        #     scene_name = scene["scene_id"]
        # scene_images = self.image_dataset.get_data_by_scene_name(scene_name)
        return dict(pcd_dict=scene)

    def __len__(self):
        return len(self.pointcloud_dataset)
