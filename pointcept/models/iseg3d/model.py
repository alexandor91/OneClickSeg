"""
懒得再写一版纯点云的了, 就这样了
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import itertools

from .positional_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D
from ..builder import MODELS, build_model
from ..losses import build_criteria, LOSSES

from pointcept.models.utils.structure import Point  # 使用 Point 类, DefaultSegmentorv2
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.utils_iseg.structure import Scene, Query


@MODELS.register_module()
class Mask3dSegmentor(nn.Module):
    def __init__(
        self,
        pcd_backbone=None,
        mask_decoder=None,
        loss=None,
        semantic=True,
        clicks_from_instance=True,
        iter_loss_weight=[0.2, 0.3, 0.4],
        mask_threshold=0.0,
    ):
        """
        interactive的segmentor, 返回没做softmax的结果
        - 直接由backbone预测seg的结果, 训练模式则self.training设为True, 其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候, __init__里加载预训练参数
        - pcd_backbone和img_backbone默认在forward过程中更新embedding
        - pcd_backbone和img_backbone需要有get_embedding方法, 获取点云和图像的embedding
        - semantic: 是否为语义分割任务
        - clicks_from_instance: 是否利用instance gt去构建click
        - iter_loss_weight: 迭代训练的loss占比
        """
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.mask_decoder = build_model(mask_decoder)
        self.loss = LOSSES.build(loss)
        self.iter_loss_weight = iter_loss_weight
        self.max_train_iter = len(iter_loss_weight)
        assert self.max_train_iter > 0, "len(iter_loss_weight) must > 0"
        self.mask_threshold = mask_threshold
        self.semantic = semantic
        self.clicks_from_instance = clicks_from_instance

        # positional encoding
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=self.embedding_dim,
            gauss_scale=1.0,
            normalize=True,
        )

    def construct_scene(self, point: Point):
        """
        构建 scene, 包括 positional encoding
        """
        scene = Scene(point)

    def refine(self, pred_last, pcd_dict, clicks):
        """
        场景点云的预测结果, 用该场景的RGBD去refine
        - pred_last: 上一次的预测
            - 'masks_heatmap': tensor, N_points x N_clicks
            - 'cls_logits': tensor, N_clicks x num_classes
        - pcd_dict: fragment_list 中的一个, 做了voxelize, ['coord', 'discrete_coord', 'index', 'offset', 'feat']
        - clicks: list, N_clicks

        Returns:
        - pred_refine: 更新后的预测
            - 'masks_heatmap': tensor, N_points x N_clicks
            - 'cls_logits': tensor, N_clicks x num_classes

        TODO:
        - 如何利用pred?
        """
        if len(clicks) == 0:
            return pred_last
        last_masks_heatmap = pred_last["masks_heatmap"]
        masks_heatmap, cls_logits = self.mask_decoder(
            pcd_dict, clicks, last_masks_heatmap
        )  # N_points x N_clicks, N_clicks x num_classes

        # new_logits = masks_heatmap @ cls_logits        # N_points x num_classes, 相当于point wise地叠加了多个mask

        # TODO: 这样做会有问题, 即如果 click 查询出的mask不能覆盖整个点云里面含有instance的部分, 那么会出现某些point的heatmap截断为0, 这样logits也全为0, 会误分类为wall;

        pred_refine = {"masks_heatmap": masks_heatmap, "cls_logits": cls_logits}

        return pred_refine

    def forward_train(self, pcd_dict, clicker, img_dict, fusion):
        """
        训练阶段, 迭代优化pred
        - scene: N_point data dict
            - feature_pyramid: features collection layer
                - len(scene.feature_pyramid.features) == self.features_num
                - scene.feature_pyramid.features[j]: (N_points_j, self.features_dims[j])
                - features sorted from coarse to fine
        """
        # 训练策略写在这里了
        # with torch.no_grad():
        # 前 self.max_train_iter-1 次, 都不用梯度回传, 只有最后一次会回传梯度
        pcd_pred = pcd_dict["pcd_pred"]  # (n, k)
        clicks = clicker.make_init_clicks()  # 获取 clicks simulation
        pred_last = {"masks_heatmap": None, "cls_logits": None}
        # TODO: 需要对clicks做组合, 使得能分割 single object; 但这么做会爆显存, 即使i取1也不行
        loss_all = None
        # for i in range(1,3):
        #     clicks_selected_list = list(itertools.combinations(clicks, i))
        #     for clicks_selected in clicks_selected_list:
        #         masks_heatmap, cls_logits, pred_refine = self.refine(pcd_pred, pcd_dict, clicks_selected)    # 优化pred
        #         clicks_dict = dict(
        #             masks_heatmap=masks_heatmap, cls_logits=cls_logits,
        #             pred_last=pcd_pred, pred_refine=pred_refine,
        #             clicks=clicks_selected, clicks_from_instance=self.clicks_from_instance
        #         )
        #         loss, info_dict = self.loss(clicks_dict, pcd_dict)
        #         if loss_all == None:
        #             loss_all = loss
        #         else:
        #             loss_all += loss

        for i in range(self.max_train_iter):
            pred_refine = self.refine(pred_last, pcd_dict, clicks)  # 优化pred
            masks_heatmap, cls_logits = (
                pred_refine["masks_heatmap"],
                pred_refine["cls_logits"],
            )
            clicks_dict = dict(
                pred_last=pred_last,
                pred_refine=pred_refine,
                clicks=clicks,
                clicks_from_instance=self.clicks_from_instance,
                mask_threshold=self.mask_threshold,
            )
            loss, info_dict = self.loss(clicks_dict, pcd_dict)
            # 记得下面返回的loss也要改
            if loss_all == None:
                loss_all = self.iter_loss_weight[i] * loss
            else:
                loss_all += self.iter_loss_weight[i] * loss
            pred_last["masks_heatmap"] = masks_heatmap.clone().detach()
            pred_last["cls_logits"] = cls_logits.clone().detach()
            clicker.update_pred(info_dict["class_tag_to_mask"])
            clicks = clicker.make_next_clicks()

        # 后面需要用 masks_heatmap 梯度, 所以这里建个新的, 截断用来算 per click mask iou
        masks = masks_heatmap.clone().detach()
        masks[masks <= self.mask_threshold] = 0
        masks[masks > 0] = 1
        masks_target = info_dict["masks_target"]
        cls_target = info_dict["cls_target"]
        cls_pred = F.softmax(cls_logits, -1).max(1)[1]
        intersection, union, target = intersection_and_union_gpu(
            cls_pred, cls_target, 20, -1
        )
        clicks_cls_iou = intersection / (union + 1e-10)
        clicks_cls_iou = clicks_cls_iou[union != 0].mean()
        clicks_masks_intersection = (masks * masks_target).sum(0)
        clicks_masks_union = ((masks + masks_target) > 0).float().sum(0)
        clicks_masks_iou = clicks_masks_intersection / clicks_masks_union
        clicks_masks_ap50 = torch.Tensor(
            [
                (clicks_masks_iou[clicks_masks_iou > 0.5]).shape[0]
                / clicks_masks_iou.shape[0]
            ]
        )

        # iter for max_train_iter
        # for i in range(self.max_train_iter-2):
        #     clicker.update_pred(pred_refine.max(1)[1].data)
        #     clicks = clicker.make_next_click()
        #     pred_refine = self.refine(pred_refine, pcd_dict, clicks)
        #     loss += self.loss(pred_refine, pcd_dict, clicks)
        # clicker.update_pred(pred_refine.max(1)[1].data)
        # clicks = clicker.make_next_click()
        # pred_refine = self.refine(pred_refine, pcd_dict, clicks)
        # seg_logits = pred_refine
        # loss += self.loss(seg_logits, pcd_dict, clicks)

        return dict(
            loss=loss_all,
            clicks_mask_loss=info_dict["clicks_mask_loss"],
            clicks_cls_loss=info_dict["clicks_cls_loss"],
            mask_loss=info_dict["mask_loss"],
            clicks_masks_iou=clicks_masks_iou.mean(),
            clicks_masks_ap50=clicks_masks_ap50,
            clicks_cls_iou=clicks_cls_iou,
        )

    def forward_eval(self, pcd_dict, clicker, img_dict, fusion):
        pcd_pred = pcd_dict["pcd_pred"]  # (n, k)
        clicks = clicker.make_init_clicks()  # 获取 clicks simulation
        pred_last = {"masks_heatmap": None, "cls_logits": None}
        pred_refine = self.refine(pred_last, pcd_dict, clicks)  # 优化pred
        masks_heatmap, cls_logits = (
            pred_refine["masks_heatmap"],
            pred_refine["cls_logits"],
        )
        clicks_dict = dict(
            pred_last=pred_last,
            pred_refine=pred_refine,
            clicks=clicks,
            clicks_from_instance=self.clicks_from_instance,
            mask_threshold=self.mask_threshold,
        )
        loss, info_dict = self.loss(clicks_dict, pcd_dict)
        return dict(
            loss=loss, seg_logits=masks_heatmap @ cls_logits, seg_logits_last=pcd_pred
        )

    def forward_test(self, pcd_dict, clicker, img_dict, fusion):
        """测试阶段, 没有真值, 随机取点并聚合mask"""
        pcd_pred = pcd_dict["pcd_pred"]  # (n, k)
        assert "fragment_idx" in pcd_dict.keys(), "must have fragment idx for tester"
        clicks = clicker.make_init_clicks(
            fragment_idx=pcd_dict["fragment_idx"],
            fragment_feature=pcd_dict["feature_maps"][-1].features,
        )  # 获取 clicks simulation
        pred_last = {"masks_heatmap": None, "cls_logits": None}
        if "pred_last" in pcd_dict.keys():
            pred_last = pcd_dict["pred_last"]
        pred_refine = self.refine(pred_last, pcd_dict, clicks)  # 优化pred
        masks_heatmap, cls_logits = (
            pred_refine["masks_heatmap"],
            pred_refine["cls_logits"],
        )
        pred_refine["masks_heatmap"][
            masks_heatmap <= self.mask_threshold
        ] = 0  # 非训练时对mask做截断
        clicks_dict = dict(
            pred_last=pred_last,
            pred_refine=pred_refine,
            clicks=clicks,
            clicks_from_instance=self.clicks_from_instance,
            mask_threshold=self.mask_threshold,
        )
        return clicks_dict

    def forward_demo(self, pcd_dict, clicks, img_dict, fusion):
        """demo, 根据给定的clicks做segment"""
        pcd_pred = pcd_dict["pcd_pred"]  # (n, k)
        pred_last = {"masks_heatmap": None, "cls_logits": None}
        if "pred_last" in pcd_dict.keys():
            pred_last = pcd_dict["pred_last"]
        pred_refine = self.refine(pred_last, pcd_dict, clicks)  # 优化pred
        masks_heatmap, cls_logits = (
            pred_refine["masks_heatmap"],
            pred_refine["cls_logits"],
        )
        pred_refine["masks_heatmap"][
            masks_heatmap <= self.mask_threshold
        ] = 0  # 非训练时对mask做截断
        clicks_dict = dict(
            pred_last=pred_last,
            pred_refine=pred_refine,
            clicks=clicks,
            clicks_from_instance=self.clicks_from_instance,
            mask_threshold=self.mask_threshold,
        )
        return clicks_dict

    def forward(self, input_dict):
        """分开写成forward_train"""
        pcd_dict = input_dict["pcd_dict"]
        clicker = input_dict["clicker"]
        img_dict = input_dict["img_dict"]
        fusion = input_dict["fusion"]

        res = self.pcd_backbone(pcd_dict)
        pcd_dict["pcd_pred"] = res["seg_logits"]  # (n, k)
        pcd_dict["feature_maps"] = res[
            "feature_maps"
        ]  # pcd embedding list, [(N_points, embedding_dims[i])]

        # train
        if self.training:
            clicker.set_state(
                "train",
                [pcd_dict],  # 这里加一层, 因为需要的是 fragment_list
                img_dict=img_dict,
                fusion=fusion,
            )
            return self.forward_train(pcd_dict, clicker, img_dict, fusion)
        # val
        elif "segment" in pcd_dict.keys():
            clicker.set_state("val", [pcd_dict], img_dict=img_dict, fusion=fusion)
            return self.forward_eval(pcd_dict, clicker, img_dict, fusion)
        # interactive demo & progressive test
        elif "clicks" in input_dict.keys():
            clicks = input_dict["clicks"]
            return self.forward_demo(pcd_dict, clicks, img_dict, fusion)
        # test
        else:
            # clicker.set_state 放到了外面
            return self.forward_test(pcd_dict, clicker, img_dict, fusion)

    # def get_scene_pe(self, scene_dict):
    #     '''
    #     获取scene的positional embedding, TODO 需要为每个分辨率单独生成 pe
    #     '''
    #     offset = scene_dict["offset"]
    #     batch = offset2batch(offset)
    #     scene_pe = []
    #     for b in range(len(offset)):
    #         batch_mask = batch == b
    #         mins  = scene_dict["grid_coord"][batch_mask].min(dim=0)[0]
    #         maxs = scene_dict["grid_coord"][batch_mask].max(dim=0)[0]
    #         pe = self.pos_enc(
    #             scene_dict["grid_coord"][batch_mask].float(),
    #             input_range=[mins, maxs]
    #         )
    #         scene_pe.append(pe)
    #     return torch.cat(scene_pe)

    # def get_clicks_index(self, clicks, scene_dict):
    #     '''
    #     获取click在scene中的index, 用 click.index在grid_coord中寻址
    #     - clicks[batch_id][class_tag]['init'] -> MyClick
    #     - scene_dict: 一般为fragment处理后的input_dict
    #     '''
    #     batch = offset2batch(scene_dict["offset"])
    #     grid_coord = torch.cat([batch.unsqueeze(-1).int(), scene_dict["grid_coord"].int()], dim=1).contiguous()
    #     for batch_id in range(len(clicks)):
    #         for class_tag in clicks[batch_id].keys():
    #             for name in clicks[batch_id][class_tag].keys():
    #                 click = clicks[batch_id][class_tag][name]
    #                 if click.index != None:
    #                     continue
    #                 mask_x = grid_coord[:,1] == click.coords[0]
    #                 mask_y = grid_coord[:,2] == click.coords[1]
    #                 mask_z = grid_coord[:,3] == click.coords[2]
    #                 mask_b = grid_coord[:,0] == click.batch_id
    #                 mask = torch.logical_and(torch.logical_and(mask_x, mask_y), torch.logical_and(mask_z, mask_b))
    #                 click.index = torch.where(mask)[0].item()

    # def get_clicks_idxs(self, clicks, N_points):
    #     '''
    #     获取click的index list
    #     TODO 区分positive 和 negative, 不同click用不同batch? 一个click分3channel, initial click, pos click, neg click
    #     - clicks:
    #         - clicks[batch_id][class_tag]['init'] -> MyClick, 见 pointcept.utils.clicker
    #         - 可用 click.index在grid_coord中寻址
    #     - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
    #     '''
    #     clicks_idx_list = []
    #     for batch_id in range(len(clicks)):
    #         for class_tag in clicks[batch_id].keys():
    #             init = clicks[batch_id][class_tag]['init']
    #             pos = clicks[batch_id][class_tag]['pos']
    #             neg = clicks[batch_id][class_tag]['neg']
    #             init, pos = init.index, pos.index
    #             if neg == None:
    #                 neg = N_points
    #             else:
    #                 neg = neg.index
    #             clicks_idx_list.extend([init, pos, neg])
    #     clicks_idxs = torch.Tensor(clicks_idx_list).long()
    #     return clicks_idxs

    # def get_clicks_feat(self, clicks_idxs, scene_embedding):
    #     '''
    #     获取click的feature
    #     - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
    #     - scene_embedding (torch.Tensor): scene pcd embedding. Shape as N_points x embedding_dim for any N_points.
    #     - clicks_embedding: (torch.Tensor), N_clicks x 3 x embedding_dim
    #     '''
    #     scene_embedding = torch.cat([scene_embedding, torch.zeros((1, scene_embedding.shape[1])).to(scene_embedding.device)])
    #     return scene_embedding[clicks_idxs].detach().clone().reshape(-1, 3, scene_embedding.shape[1])

    # def get_clicks_pe(self, clicks_idxs, scene_pe):
    #     '''
    #     获取click的positional embedding
    #     - clicks_idxs: 1d tensor, [N_clicks x (init, pos, neg)]
    #     - scene_pe (torch.Tensor): scene positional embedding. Shape as N_points x embedding_dim for any N_points.
    #     - clicks_pe: (torch.Tensor), N_clicks x 3 x embedding_dim
    #     '''
    #     scene_pe = torch.cat([scene_pe, torch.zeros((1, scene_pe.shape[1])).to(scene_pe.device)])
    #     return scene_pe[clicks_idxs].detach().clone().reshape(-1, 3, scene_pe.shape[1])

    # def get_clicks_batch(self, clicks):
    #     '''获取click属于哪个batch'''
    #     clicks_batch = []
    #     for batch_id in range(len(clicks)):
    #         clicks_batch.extend([batch_id for _ in clicks[batch_id].keys()])
    #     return torch.Tensor(clicks_batch).int()
