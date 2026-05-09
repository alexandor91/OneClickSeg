"""
agile3d v1m1, 直接照搬 mask3d select query
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import itertools
from addict import Dict

from ..builder import MODELS, build_model
from ..losses import LOSSES
from ..mask3d.positional_embedding import PositionEmbeddingCoordsSine

from pointcept.models.utils.structure import Point  # 使用 Point 类, DefaultSegmentorv2
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


@MODELS.register_module("Agile3d-v1m1")
class Agile3d(nn.Module):
    def __init__(
        self,
        pcd_backbone=None,
        mask_decoder=None,
        fused_backbone=False,
        on_segment=False,
        loss=None,
        matcher_cfg=None,
        aux=True,
        features_dims=(256, 256, 128, 96, 96),
        num_query=200,
        query_type="learn",
        mask_threshold=0.5,
        topk_per_scene=200,
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
    ):
        """
        - pcd_backbone: 点云backbone
        - mask_decoder: mask decoder
        - fused_backbone: 是否冻住 backbone 参数, 不参与梯度更新
        - on_segment: 是否在 sem seg 结果上做实例分割
        - loss: 损失函数
        - matcher_cfg: matcher 配置参数
        - aux: 是否采用 aux loss
        - features_dims: 多尺度特征维度
        - num_query: 查询点数
        - query_type: 查询点类型, "learn" or "sample"
        - mask_threshold: 测试时 mask 截断 threshold
        - topk_per_scene: 每个场景中保留的预测实例数, 用于 soft grouping
        - semantic_ignore: 忽略的 semantic id
        - instance_ignore: 忽略的 instance id
        - semantic_background: 背景类, 用于测试时去除背景类
        """
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        if fused_backbone:
            for p in self.pcd_backbone.parameters():
                p.requires_grad = False
        self.mask_decoder = build_model(mask_decoder)
        self.on_segment = on_segment
        self.loss = LOSSES.build(loss)
        self.matcher = HungarianMatcher(**matcher_cfg)
        self.aux = aux
        self.features_dims = features_dims
        self.num_query = num_query
        self.query_type = query_type
        self.mask_threshold = mask_threshold
        self.topk_per_scene = topk_per_scene
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background
        self.features_num = len(features_dims)
        assert (
            self.features_num == self.mask_decoder.features_num
        ), f"features_num {self.features_num} must be same as mask_decoder.features_num {self.mask_decoder.features_num}"
        self.num_classes = self.mask_decoder.num_classes
        self.embedding_dim = self.mask_decoder.embedding_dim

        # scene
        self.mask_feat_proj = nn.Linear(features_dims[-1], self.embedding_dim)
        self.scene_norm = nn.LayerNorm(
            self.embedding_dim
        )  # 在 in_proj 后加一个 ln, 对齐特征
        self.in_proj_scene_layers = nn.ModuleList()
        for i in range(self.features_num):
            in_proj = nn.Linear(features_dims[i], self.embedding_dim)
            self.in_proj_scene_layers.append(in_proj)

        # query
        assert self.query_type in ["learn", "sample", "zero"]
        if self.query_type == "learn":
            # learnable, both feat and pe
            self.query_feat = nn.Embedding(num_query, self.embedding_dim)
            self.query_pe = nn.Embedding(num_query, self.embedding_dim)
        elif self.query_type == "sample":
            # sample, feat and pe
            self.in_proj_query_feat = nn.Linear(
                self.features_dims[-1], self.embedding_dim
            )
        #     self.in_proj_query_pe = nn.Linear(self.features_dims[-1], self.embedding_dim)
        # elif self.query_type == "zero":
        #     # zero feat, only sample pe
        #     self.in_proj_query_pe = nn.Linear(self.features_dims[-1], self.embedding_dim)
        # positional encoding
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=self.embedding_dim,
            gauss_scale=1.0,
            normalize=True,
        )

    def get_scene_pe(self, coord_list, batch_list):
        """
        获取 positional encoding
        - coord_list: (features_num, N_point, 3)
        - batch_list: (features_num, N_point,)
        """
        bs = batch_list[-1].max().item() + 1
        # 使用最高分辨率坐标, 作为归一化的范围
        mins_list = [
            coord_list[-1][batch_list[-1] == b].min(dim=0)[0] for b in range(bs)
        ]
        maxs_list = [
            coord_list[-1][batch_list[-1] == b].max(dim=0)[0] for b in range(bs)
        ]
        pe_list = []
        for coord, batch in zip(coord_list, batch_list):
            pe = []
            for b in range(bs):
                mask = batch == b
                mins = mins_list[b]
                maxs = maxs_list[b]
                pe.append(self.pos_enc(coord[mask].float(), input_range=[mins, maxs]))
            pe_list.append(torch.cat(pe))
        return pe_list

    def find_nearest_point(self, source, query, source_batch, query_batch):
        """
        获取 points 在 scene 中的最近点, 返回 index (offset 对应的 batch 内的, 不是全局index)
        - source: (N_point, 3)
        - query: (N_query, 3)
        """
        assert (
            source_batch.max().item() == query_batch.max().item()
        ), "source_batch and query_batch must have the same shape"
        bs = source_batch.max().item() + 1
        dist = torch.cdist(source, query)
        matched_idx = torch.zeros(query.shape[0], dtype=torch.long)
        for i in range(bs):
            mask_s = source_batch == i
            mask_q = query_batch == i
            matched_idx[mask_q] = dist[mask_s, mask_q].argmin(dim=0)
        return matched_idx

    def construct_scene(self, pcd_dict):
        """
        构建 scene
        - scene: scene data dict
            - semantic: (N_point[-1],), gt semantic id
            - instance: (N_point[-1],), gt instance id
            - feature_pyramid: features collection layer
                - x_list(): return x_list, x in ['feat', 'coord', 'batch']
                - downsample(): downsample feat
                - features sorted from coarse to fine
            - feat_list: list, feat_list[j].shape == (N_point[j], features_dims[j])
            - coord_list: list, coord_list[j].shape == (N_point[j], 3)
            - batch_list: list, batch_list[j].shape == (N_point[j],)
            - pe_list: list, pe_list[j].shape == (N_point[j], embedding_dim)
        """
        scene = Scene()
        scene.semantic = pcd_dict["segment"]
        scene.instance = pcd_dict["instance"]
        scene.feature_pyramid = self.pcd_backbone.feature_pyramid
        scene.feat_list = scene.feature_pyramid.feat_list()
        scene.coord_list = scene.feature_pyramid.coord_list()
        scene.batch_list = scene.feature_pyramid.batch_list()
        scene.pe_list = self.get_scene_pe(scene.coord_list, scene.batch_list)
        return scene

    def construct_query(self, pcd_dict, scene: Scene):
        """
        构建 query
        - query: query data dict
            - feat: (N_query, query_embedding_dim)
            - pe: positional encoding, (N_query, embedding_dim)
            - batch: (N_query,)
        """
        query = Query()
        bs = scene.batch_list[-1].max().item() + 1
        if self.query_type == "learn":
            # 将 self.latent_query 复制 b 次
            query.feat = self.query_feat.weight.repeat(bs, 1).to(
                scene.feat_list[-1].device
            )
            query.pe = self.query_pe.weight.repeat(bs, 1).to(scene.feat_list[-1].device)
            query.batch = torch.arange(bs, device=query.feat.device).repeat_interleave(
                self.num_query
            )
        elif self.query_type == "sample":
            idx = torch.where(pcd_dict["sampled_mask"])[0]
            gt_ins_id = self.scene.instance[idx]
            # TODO 将 semantic background 纳入训练
            # 去掉 semantic background
            idx = idx[gt_ins_id != self.instance_ignore]
            if self.training and len(idx) > self.num_query:
                # 如果训练时采样点数大于 num_query, 则随机选部分
                # TODO 有什么办法改变这个行为? 只是为了确保不超显存? 太不均匀
                # TODO 在同一 id 内随机采样
                # 好像跟采样方式有关，agile3d能保证每个batch的query point分散开，且类之间均匀
                shuffle_idx = torch.randperm(len(idx))
                idx = idx[shuffle_idx][: self.num_query]
            query.idx = idx
            query.gt_ins_id = self.scene.instance[idx]  # 重新计算
            query.feat = scene.feat_list[-1][idx]
            query.pe = scene.pe_list[-1][idx]
            query.batch = scene.batch_list[-1][idx]
        return query

    def add_new_query(self, new_query):
        """
        添加新的 query
        """
        # TODO 最后记得做 in_proj
        # self.query.feat[-N_query_new:] = self.in_proj_query(self.query.feat[-N_query_new:])   # 为新query单独做 in_proj
        pass

    def in_projection(self, scene, query):
        """
        - scene: list, scene[j] == (N_point[j], features_dims[j])
        - query: (N_query, query_embedding_dim)
        - scene_out: list, scene_out[j] == (N_point[j], embedding_dim)
        - query_out: (N_query, embedding_dim)
        - scene_mask_feat: (N_point[-1], embedding_dim), 作为 source 在 head 处用于计算 mask
        """
        if self.query_type == "sample":
            query = self.in_proj_query_feat(query)
        scene_mask_feat = self.mask_feat_proj(scene[-1])
        scene_mask_feat = self.scene_norm(scene_mask_feat)
        scene_out = []
        for j in range(len(scene)):
            feat = scene[j]  # (N_point[j], self.features_dims[j])
            feat = self.in_proj_scene_layers[j](
                feat.float()
            )  # (N_point[j], embedding_dim)
            feat = self.scene_norm(feat)
            scene_out.append(feat)
        return scene_out, query, scene_mask_feat

    def refine(self, pred_last):
        """
        细化场景点云的预测结果
        - pred_last: 上一次的预测
            - 'masks_logits': (N_query_last, N_point[-1])
            - 'cls_logits': (N_query_last, num_classes)\n
        Returns:
        - preds: 更新后的预测, list(dict), 如果采用 aux loss, 则包含每层 decoder 的输出, 用于监督, 否则只返回最后一层
            - 'masks_logits': N_query x N_point[j // num_decoders]
            - 'cls_logits': N_query x num_classes
        Tips:
        - 如何利用pred_last? 使用 float type attention mask
        """
        preds = self.mask_decoder(self.scene, self.query, pred_last)
        if not self.aux:
            preds = [preds.pop()]
        return preds

    def match_pred_target(self, pred, target):
        """构建与匹配的监督信号, 用于计算 loss;\n
        - pred, target is the result of get_pred() and get_target()
        - add matched_idx to pred and target, shape: (N_matched,)"""
        if self.query_type == "learn":
            pred["matched_idx"], target["matched_idx"] = self.matcher(pred, target)
        elif self.query_type == "sample":
            # pred["matched_idx"], target["matched_idx"] = self.matcher(pred, target)
            pred_matched_ins_id = self.query.gt_ins_id
            assert torch.isin(
                pred_matched_ins_id, target["ins_id"]
            ).all(), "pred_matched_ins_id must be in target['ins_id']"
            pred["matched_idx"] = torch.arange(
                pred_matched_ins_id.shape[0], device=pred_matched_ins_id.device
            )
            sorted_target_ins_id, sort_indices = torch.sort(target["ins_id"])
            # 快速计算把 a 插入 b 中的位置, 相等则在前插
            buckets = torch.bucketize(pred_matched_ins_id, sorted_target_ins_id)
            target["matched_idx"] = sort_indices[buckets]
        return pred, target

    def post_process(self, pred):
        """
        后处理, 将 pred dict 转换为 互相排斥的 panoptic 结果, 同时做 soft grouping (arXiv:2203.01509)
        注意shape尚未被 get_pred() 改变
        - pred: dict, 预测结果
            - 'masks_logits': (N_query, N_point[-1])
            - 'cls_logits': (N_query, num_classes)
            - other attributes with (N_query, ...)

        Returns: pred dict
        - 'masks': (N_pred_instance, N_point[-1]), int tensor binary mask, for eval & test
        - 'masks_logits': (N_pred_instance, N_point[-1]), as logits, for loss func
        - 'cls_logits': (N_pred_instance, num_classes)
        - 'scores': (N_pred_instance,), 每个 instance 的得分
        - other attributes with (N_pred_instance, ...)
        """
        # 不能直接截断, 会出现某些point的heatmap截断为全0, argmax默认属于 query 0
        # 解决: 加一个 ignore filter mask
        masks_logits = pred["masks_logits"]
        masks = torch.cat(
            [
                torch.ones_like(masks_logits[:1, :]) * self.mask_threshold,
                masks_logits.sigmoid(),
            ],
            dim=0,
        ).contiguous()
        # TODO 把背景类也作为instance纳入训练, 测试时通过 semantic proto 做 query 来去除, 不关心的点为 -1
        pred_instance = (
            mask_to_id(masks) - 1
        )  # 预测的instance id, 互相排斥的 panoptic 结果
        pred_instance[pred_instance == -1] = self.instance_ignore
        unique_pred_id = unique_id(
            pred_instance, self.instance_ignore
        )  # filter 后, 有效的 query idx
        masks = id_to_mask(pred_instance, unique_pred_id)

        for key in pred.keys() - {"matched_idx"}:
            pred[key] = pred[key][unique_pred_id]
        pred["masks"] = masks
        pred["masks_logits"] = (
            pred["masks_logits"] * masks
        )  # 相当于用 mask_threshold 阈值截断 masks_logits

        pred = softgroup_post_process(pred, self.topk_per_scene)
        return pred

    def forward(self, pcd_dict):
        """one time forward"""
        # TODO train on segment
        seg_logits = self.pcd_backbone(pcd_dict)  # (n, k)
        pcd_dict["pcd_pred"] = seg_logits
        self.scene = self.construct_scene(pcd_dict)
        self.query = self.construct_query(pcd_dict, self.scene)
        (
            self.scene.feat_list,
            self.query.feat,
            self.scene.mask_feat,
        ) = self.in_projection(self.scene.feat_list, self.query.feat)

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
