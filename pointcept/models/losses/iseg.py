"""
Interactive Losses

Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict

from .builder import LOSSES, build_criteria
from pointcept.models.utils import offset2batch
from pointcept.utils.misc import intersection_and_union_gpu


@LOSSES.register_module()
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", logits=False, loss_weight=1.0):
        super(BinaryCrossEntropyLoss, self).__init__()
        if logits:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.loss = torch.nn.BCELoss(reduction=reduction)
        self.loss_weight = loss_weight

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        loss = self.loss(pred, target)
        return loss * self.loss_weight


@LOSSES.register_module()
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        exponent: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N_points, N_masks]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N_masks,] if 'none'
        logits: if True, predict is unnormalized
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(
        self, smooth=1, exponent=2, reduction="mean", logits=False, loss_weight=1.0
    ):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.logits = logits
        self.loss_weight = loss_weight

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        if self.logits:
            predict = predict.sigmoid()

        num = torch.sum(torch.mul(predict, target), dim=0) * 2 + self.smooth
        den = (
            torch.sum(predict.pow(self.exponent) + target.pow(self.exponent), dim=0)
            + self.smooth
        )

        loss = 1 - num / den

        if self.reduction == "mean":
            return self.loss_weight * loss.mean()
        elif self.reduction == "sum":
            return self.loss_weight * loss.sum()
        elif self.reduction == "none":
            return self.loss_weight * loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


@LOSSES.register_module()
class InsSegLoss(nn.Module):
    def __init__(self, cls_loss_cfg=None, mask_loss_cfg=None):
        super(InsSegLoss, self).__init__()
        self.cls_loss = build_criteria(cls_loss_cfg)
        self.mask_loss = build_criteria(mask_loss_cfg)

    def select_matched_pred_target(self, pred: dict, target: dict):
        """按照匹配的idx, slice所有属性, idx 都是 dim 0 的索引\n
        返回新的字典(不影响用原字典做ins seg测试)"""
        idx_pred, idx_target = pred["matched_idx"], target["matched_idx"]
        pred_new, target_new = {}, {}
        for k in pred.keys() - {"matched_idx"}:
            pred_new[k] = pred[k][idx_pred]
        for k in target.keys() - {"matched_idx", "is_testset"}:
            target_new[k] = target[k][idx_target]
        return pred_new, target_new

    def compute_info(self, pred, target):
        """计算每个 mini batch 需要的 log, 必须是浮点数
        - pred, target: 匹配并且slice后的, 可以按 idx 对应, 个数需一致"""
        # 后面需要用 masks_logits 梯度, 所以这里建个新的, 截断用来算 per click mask iou
        masks_pred = pred["masks_logits"].clone().detach().sigmoid()
        masks_target = target["masks"]
        cls_pred = pred["cls"]
        cls_target = target["cls"]

        cls_precision = (cls_pred == cls_target).float().sum() / cls_pred.shape[0]

        # 计算每个实例的IoU，先计算每个实例的交集和并集
        masks_intersection = (masks_pred * masks_target).sum(
            1
        )  # 按点求和，在第1维(点维度)求和
        pred_areas = masks_pred.sum(1)  # N_instances
        target_areas = masks_target.sum(1)  # N_instances
        masks_union = pred_areas + target_areas - masks_intersection  # 正确计算并集

        # 避免除零错误
        masks_iou = (masks_intersection / (masks_union + 1e-6)).mean()

        return Dict(cls_precision=cls_precision, masks_iou=masks_iou)

    def forward(self, preds, target):
        """
        Arguments
        - preds: list of pred dict
            - masks_logits: [N_pred, N_points], pred masks logits
            - cls_logits: [N_pred, num_classes], pred cls logits
            - cls_prob: [N_pred, num_classes], pred cls prob, for matcher
            - cls: [N_pred,], pred semantic class
            - batch: [N_pred,], pred batch id
            - matched_idx: [N_matched,], pred's matched target idx, 只在最后一个dict中
        - target: target dict
            - masks: [N_target, N_points], binary mask
            - cls: [N_target,], target semantic class
            - batch: [N_target,], target batch id
            - matched_idx: [N_matched,], target's matched pred idx
        """
        # 按匹配的 idx 筛选, 后计算 loss
        loss = torch.tensor(0.0, device=preds[0]["masks_logits"].device)
        pred_idx, target_idx = preds[-1]["matched_idx"], target["matched_idx"]
        target_new = {}
        for k in target.keys() - {"matched_idx", "is_testset"}:
            target_new[k] = target[k][target_idx]
        for pred in preds:
            pred_new = {}
            for k in pred.keys() - {"matched_idx"}:
                pred_new[k] = pred[k][pred_idx]
            mask_loss = self.mask_loss(
                pred_new["masks_logits"], target_new["masks"]
            )  # binary ce, 用focal loss 来算 bce
            cls_loss = self.cls_loss(pred_new["cls_logits"], target_new["cls"].long())
            loss += mask_loss + cls_loss
        loss = loss / len(preds)
        # 打印的信息, 必须是一个浮点数
        info_dict = self.compute_info(pred_new, target_new)
        info_dict = Dict(mask_loss=mask_loss, cls_loss=cls_loss, **info_dict)
        # loss 做反向传播(engines/train.py), 其他信息打印log(engines/hooks/misc.py)
        return loss, info_dict


@LOSSES.register_module()
class Agile3DLoss(nn.Module):
    def __init__(
        self,
        clicks_mask_loss_cfg=None,
        clicks_cls_loss_cfg=None,
        mask_loss_cfg=None,
    ):
        super(Agile3DLoss, self).__init__()
        self.clicks_mask_loss = build_criteria(clicks_mask_loss_cfg)
        self.clicks_cls_loss = build_criteria(clicks_cls_loss_cfg)
        self.mask_loss = build_criteria(mask_loss_cfg)

    def get_clicks_target(self, clicks, target, instance, pred_masks, batch):
        """
        构建以click为中心的监督信号
        - clicks: N_clicks, 详见 clicker
        - target: tensor on gpu, shape as (N_points,)
        - instance: tensor on gpu, shape as (N_points,), masks target
        - pred_masks: N_points x N_clicks, False or True
        - batch: which batch the point belongs to , shape as (N_points,)

        Return:
        - masks_target: N_points x N_clicks, 一个点不一定只对应一个click
        - cls_target: N_clicks
        - class_tag_to_mask: batch_id, class_tag -> 1d mask
        """
        class_tag_to_mask = {}
        masks_target = []
        index = []
        for i in range(len(clicks)):
            for j in clicks[i].keys():
                click = clicks[i][j]["init"]
                index.append(click.index)
                class_tag = click.class_tag
                batch_id = click.batch_id
                mask = torch.zeros((target.shape[0],)).float()
                mask[((instance == class_tag) * (batch == batch_id)).bool()] = 1.0
                masks_target.append(mask)
                if batch_id not in class_tag_to_mask.keys():
                    class_tag_to_mask[batch_id] = {}
                class_tag_to_mask[batch_id][class_tag] = pred_masks[:, i]

        masks_target = torch.stack(masks_target, dim=1).to(target.device)
        index = torch.Tensor(index).long()
        cls_target = target[index]

        return masks_target, cls_target, class_tag_to_mask

    def forward(self, clicks_dict, pcd_dict):
        """
        Arguments
        - masks_heatmap: 场景中点属于哪个click对应的mask, 归一化后的, N_points x N_clicks
        - cls_logits: click点对应的类别, N_clicks x num_classes
        - pred_last: 优化前的prediction, N_points x num_classes
        - pred_refine: 优化后的prediction, N_points x num_classes
        - clicks: 点击list, N_clicks
        - clicks_from_instance: 是否通过instance来构建click
        - pcd_dict:
        """
        clicks = clicks_dict["clicks"]
        pred_last = clicks_dict["pred_last"]
        pred_refine = clicks_dict["pred_refine"]
        mask_threshold = clicks_dict["mask_threshold"]
        last_masks_heatmap, last_cls_logits = (
            pred_last["masks_heatmap"],
            pred_last["cls_logits"],
        )
        masks_heatmap, cls_logits = (
            pred_refine["masks_heatmap"],
            pred_refine["cls_logits"],
        )
        pred_refine = masks_heatmap @ cls_logits
        clicks_from_instance = clicks_dict["clicks_from_instance"]

        target = pcd_dict["segment"]
        grid_coord = pcd_dict["grid_coord"]
        batch = offset2batch(pcd_dict["offset"])

        masks_target, cls_target, class_tag_to_mask = self.get_clicks_target(
            clicks,
            target,
            pcd_dict["instance"] if clicks_from_instance else target,
            masks_heatmap > mask_threshold,
            batch,
        )  # 可改这里成instance监督

        clicks_mask_loss = self.clicks_mask_loss(
            masks_heatmap, masks_target
        )  # binary ce, 用focal loss 来算 bce
        clicks_cls_loss = self.clicks_cls_loss(
            cls_logits, cls_target
        )  # reduction="mean", click之间归一化
        mask_loss = self.mask_loss(pred_refine, target)
        loss = clicks_mask_loss + clicks_cls_loss + mask_loss
        # distance = torch.ones((grid_coord.shape[0],)).cuda().to(grid_coord.device)         # 保证在同一个device上
        # distance = distance / grid_coord.shape[0]
        # # TODO 批量处理
        # for click in clicks:
        #     # 需要按照batch内部处理
        #     # 保证mask与grid_coord.shape[0]一致
        #     batch_mask = batch == click.batch_id
        #     mask = batch_mask
        #     # mask = target == click.class_tag
        #     # mask = torch.logical_and(mask, batch_mask).bool()
        #     distance_to_click = ((grid_coord[mask] - click.coords.to(grid_coord.device)) ** 2).sum(1)
        #     # distance[mask] += F.softmax(distance_to_click.float(), -1)       # 归一化, 和是1, 代替取均值了
        #     distance[mask] += distance_to_click
        # distance = F.softmax(distance.float(), -1).view(1,-1)       # 归一化, 和是1, 代替取均值了
        # # distance = distance.float().view(1,-1)
        # # loss = self.ce_loss(pred_refine, target) + self.dice_loss(pred_refine, target)
        # loss = self.ce_loss(pred_refine, target).view(-1,1)
        # loss = distance @ loss
        # loss = self.ce_loss(pred_refine, target)

        info_dict = dict(
            clicks_mask_loss=clicks_mask_loss,
            clicks_cls_loss=clicks_cls_loss,
            mask_loss=mask_loss,
            masks_target=masks_target,
            cls_target=cls_target,
            class_tag_to_mask=class_tag_to_mask,
        )
        return loss, info_dict
