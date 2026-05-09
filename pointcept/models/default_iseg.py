import torch.nn as nn
import torch_scatter

import os
import torch
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


# @MODELS.register_module()
# class DefaultSegmentorV2(nn.Module):
#     def __init__(
#         self,
#         num_classes,
#         backbone_out_channels,
#         backbone=None,
#         criteria=None,
#     ):
#         super().__init__()
#         self.seg_head = (
#             nn.Linear(backbone_out_channels, num_classes)
#             if num_classes > 0
#             else nn.Identity()
#         )
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)

#     def forward(self, input_dict):
#         point = Point(input_dict)
#         point = self.backbone(point)
#         # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
#         # TODO: remove this part after make all backbone return Point only.
#         if isinstance(point, Point):
#             feat = point.feat
#         else:
#             feat = point
#         seg_logits = self.seg_head(feat)
#         # train
#         if self.training:
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss)
#         # eval
#         elif "segment" in input_dict.keys():
#             loss = self.criteria(seg_logits, input_dict["segment"])
#             return dict(loss=loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultISegEncoder(nn.Module):
    def __init__(
        self, backbone=None, ckpt_path=None, return_fpn=True, requires_grad=True
    ):
        """
        默认的iseg encoder, 经由 default.DefaultSegmentorV2 预训练 sem seg, 返回没做softmax的logits
        - backbone: backbone的配置
        - ckpt_path: 预训练的ckpt路径
        - return_fpn: 是否返回 feature pyramid
        - requires_grad: 是否冻住参数
        """
        super().__init__()
        self.return_fpn = return_fpn
        self.backbone = build_model(backbone)

        if ckpt_path is not None:
            # 加载ckpt的state_dict
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path)
            else:
                raise RuntimeError("=> No checkpoint found at '{}'".format(ckpt_path))
            state_dict = checkpoint["state_dict"]
            # 清洗, 去掉分布式训练注册名自动加的 module.
            new_state_dict = OrderedDict()
            for name, value in state_dict.items():
                if name.startswith("module."):
                    name = name[7:]  # module.xxx.xxx -> xxx.xxx
                new_state_dict[name] = value
            self.load_state_dict(
                new_state_dict, strict=False
            )  # default.DefaultSegmentorV2 sem seg 预训练时可能会多出 seg head, 所以 strict=False

        # 冻住所有参数
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        point = Point(input_dict)
        if self.return_fpn:
            point, feature_maps = self.backbone(point)
        else:
            point = self.backbone(point)
            feature_maps = []

        return dict(point=point, feature_maps=feature_maps)


@MODELS.register_module()
class DefaultInteractiveSegmentor(nn.Module):
    def __init__(
        self, pcd_backbone, img_backbone=None, mask_decoder=None, criteria=None
    ):
        """
        默认的interactive的segmentor，返回没做softmax的结果
        - 直接由backbone预测seg的结果，训练模式则self.training设为True，其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候，__init__里加载预训练参数
        - pcd_backbone和img_backbone需要有encode方法，做点云和图像的encode
        """
        super().__init__()
        self.pcd_only = img_backbone is None
        self.pcd_backbone = build_model(pcd_backbone)
        self.img_backbone = build_model(img_backbone)
        self.mask_decoder = build_model(mask_decoder)
        self.criteria = build_criteria(criteria)

    def forward_pcd(self, pcd_dict):
        """仅点云推理"""
        return self.pcd_backbone(pcd_dict)

    def forward_img(self, img_dict):
        """仅图片推理"""
        return self.img_backbone(img_dict)

    def refine(self, pred, pcd_dict, clicks, img_dict=None, fusion=None):
        """
        场景点云的预测结果, 用该场景的RGBD去refine
        - pred: 纯pcd的预测结果
        - pcd_dict: fragment_list 中的一个, 做了voxelize, ['coord', 'discrete_coord', 'index', 'offset', 'feat']
        - fusion: ['sampled_points', 'sampled_colors', 'sampled_index', 'image_projected'], 是直接对未voxelize的point做fusion
        """
        raise NotImplementedError

    def forward(self, input_dict):
        raise NotImplementedError
