"""
agile3d v1m1, 直接照搬 mask3d select query
"""

import torch
from torch import nn
from addict import Dict
import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from ..losses import LOSSES
from ..builder import MODELS
from ..sparse_unet.spconv_unet_v1m1_base import SpUNetBase

from pointcept.utils_iseg.structure import Scene, Query
from pointcept.utils_iseg.matcher import HungarianMatcher
from pointcept.utils_iseg.ins_seg import (
    mask_to_id,
    id_to_mask,
    unique_id,
    get_pred,
    get_target,
    softgroup_post_process,
)


@MODELS.register_module("InterObject3D-v1m1")
class InterObject3D(SpUNetBase):
    def __init__(
        self,
        loss=None,
        num_query=200,
        iterative=False,
        mask_threshold=0.5,
        topk_per_scene=200,
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
        **kwargs,
    ):
        """
        - loss: 损失函数
        - num_query: 查询点数
        - mask_threshold: 测试时 mask 截断 threshold
        - topk_per_scene: 每个场景中保留的预测实例数, 用于 soft grouping
        - semantic_ignore: 忽略的 semantic id
        - instance_ignore: 忽略的 instance id
        - semantic_background: 背景类, 用于测试时去除背景类
        """
        super().__init__(**kwargs)
        self.loss = LOSSES.build(loss)
        self.num_query = num_query
        # TODO iterative training pipeline
        self.iterative = iterative
        self.mask_threshold = mask_threshold
        self.topk_per_scene = topk_per_scene
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background

    def construct_scene(self, pcd_dict):
        """
        构建 scene
        - scene: scene data dict
            - semantic: (N_point,), gt semantic id
            - instance: (N_point,), gt instance id
            - coord: (N_point, 3)
            - feat: (N_point, C)
            - batch: (N_point,)
        """
        scene = Scene()
        scene.semantic = pcd_dict["segment"]
        scene.instance = pcd_dict["instance"]
        scene.coord = pcd_dict["grid_coord"]
        scene.feat = pcd_dict["feat"]
        scene.batch = pcd_dict["batch"]
        return scene

    def construct_query(self, pcd_dict, scene):
        """
        构建 query
        - query: query data dict
            - gt_ins_id: (N_query,)
            - binary_mask: (N_query, N_point)
            - batch: (N_query,)
        """
        query = Query()
        # 将query按照instance id构建成binary map, 一个query对应多个点
        sampled_mask = pcd_dict["sampled_mask"]
        sampled_ins = torch.ones_like(scene.instance) * self.instance_ignore
        sampled_ins[sampled_mask] = scene.instance[sampled_mask]
        query.gt_ins_id = unique_id(sampled_ins, self.instance_ignore)
        query.binary_mask = id_to_mask(sampled_ins, query.gt_ins_id)
        query.batch = torch.ones_like(query.gt_ins_id)
        for i in range(query.gt_ins_id.shape[0]):
            query.batch[i] = scene.batch[query.binary_mask[i]][0].item()
        return query

    def forward_unet(self, input_dict):
        """unet forward"""
        grid_coord = input_dict["coord"]
        feat = input_dict["feat"]
        batch = input_dict["batch"]

        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        if not self.cls_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        return x.features

    def refine(self, pred_last=None):
        """refine"""
        # 很难评价, 原作里是一个 d=2 的 logit 取 max, 这里用 d=1 的 logit 看看
        mask_logits = []
        for i in range(self.query.gt_ins_id.shape[0]):
            input_dict = {
                "coord": self.scene.coord,
                "feat": torch.cat(
                    (self.scene.feat, self.query.binary_mask[i].unsqueeze(1)), dim=1
                ),
                "batch": self.scene.batch,
            }
            mask_logits.append(self.forward_unet(input_dict))
        mask_logits = torch.stack(mask_logits, dim=0)  # (N_query, N_point)
        return {
            "masks_logits": mask_logits,
        }

    def match_pred_target(self, pred, target):
        """构建与匹配的监督信号, 用于计算 loss;\n
        - pred, target is the result of get_pred() and get_target()
        - add "matched_idx" to pred and target, shape: (N_matched,)"""
        pred_matched_ins_id = self.query.gt_ins_id
        pred["matched_idx"] = torch.arange(
            pred_matched_ins_id.shape[0], device=pred_matched_ins_id.device
        )
        sorted_target_ins_id, sort_indices = torch.sort(target["ins_id"])
        # 快速计算把 a 插入 b 中的位置, 相等则在前插
        buckets = torch.bucketize(pred_matched_ins_id, sorted_target_ins_id)
        target["matched_idx"] = sort_indices[buckets]
        return pred, target

    def forward(self, pcd_dict):
        """one time forward"""
        self.scene = self.construct_scene(pcd_dict)
        self.query = self.construct_query(pcd_dict, self.scene)

        pred_last = {
            "masks_logits": torch.ones(
                (0, pcd_dict["coord"].shape[0]),
                device=pcd_dict["coord"].device,
                dtype=pcd_dict["coord"].dtype,
            ),
            "cls_logits": torch.zeros(
                (0, self.num_classes),
                device=pcd_dict["coord"].device,
                dtype=pcd_dict["coord"].dtype,
            ),
        }
        preds = self.refine(pred_last)  # 优化pred

        # 改变成合适测试的shape, 方便计算 metric
        preds[-1] = get_pred(preds[-1], self.query)
        target = get_target(self.scene, self.instance_ignore)

        # test dataset, target is empty, do post process
        if target["is_testset"]:
            preds[-1] = self.post_process(preds[-1])
            return Dict(pred=preds.pop(), target=target)
        # train & eval dataset, 返回的信息打印 log(engines/hooks/misc.py)
        # 计算 loss
        preds[-1], target = self.match_pred_target(preds[-1], target)
        loss, info_dict = self.loss(preds, target)

        if self.training:
            return Dict(loss=loss, **info_dict)
        else:
            # return pred target for evaluation
            return Dict(pred=preds.pop(), target=target, loss=loss, **info_dict)
