"""
matcher, modified from mask3d, detr, 3detr
https://github.com/JonasSchult/Mask3D/blob/main/models/matcher.py
https://github.com/facebookresearch/detr/blob/master/models/matcher.py

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast


def sigmoid_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example
                (0 for the negative class and 1 for the positive class).
        targets: A float tensor with the same shape[1] as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)  # 两两的相交的面积
    denominator = (
        inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    )  # 广播, 两两的面积和
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


sigmoid_dice_loss_jit = torch.jit.script(
    sigmoid_dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example, as logits
        targets: A float tensor with the same shape[1] as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


# linear_sum_assignment using costmap
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_focal: float = 1.0,
        cost_dice: float = 1.0,
        instance_ignore: int = -1,
    ):
        """
        for matching pred instances and gt instances
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice
        self.instance_ignore = instance_ignore

        assert (
            cost_class != 0 or cost_focal != 0 or cost_dice != 0
        ), "all costs cannot be 0"

    @torch.no_grad()
    def forward(self, pred, target):
        """Performs the matching

        Params:
        - pred: This is a dict that contains at least these entries:
            - "masks_logits": [Q, N], the predicted masks logits
            - "cls_prob": [Q, num_classes], the classification probability
            - "batch": [Q,], the batch id of each pred

        - target: This is a dict containing:
            - "masks": [P, N], the target masks
            - "cls": [P,], the class labels
            - "batch": [P,], the batch id of each gt

        Returns:
        - idx_pred: [M,], the indices of the selected predictions (in order)
        - idx_target: [M,], the indices of the corresponding selected targets (in order)
        """
        P = target["masks"].shape[0]
        Q = pred["cls_prob"].shape[0]
        out_prob = pred["cls_prob"]
        tgt_ids = target["cls"].clone().long()

        # Compute the classification cost
        filter_ignore = tgt_ids == self.instance_ignore
        tgt_ids[filter_ignore] = 0
        cost_class = -out_prob[:, tgt_ids]
        cost_class[:, filter_ignore] = -1.0

        out_mask = pred["masks_logits"]
        tgt_mask = target["masks"]
        with autocast(enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            cost_focal = sigmoid_ce_loss_jit(out_mask, tgt_mask)
            cost_dice = sigmoid_dice_loss_jit(out_mask, tgt_mask)

        # Final cost matrix
        C = (
            self.cost_focal * cost_focal
            + self.cost_class * cost_class
            + self.cost_dice * cost_dice
        )
        C = C.reshape(Q, P).cpu().numpy()  # [Q, P]
        # Add large cost to prevent cross-batch matching, with torch broadcasting
        batch_not_match = pred["batch"].unsqueeze(1) != target["batch"].unsqueeze(
            0
        )  # [Q, P]
        batch_cost = batch_not_match.float().cpu().numpy() * 1e6
        C = C + batch_cost

        # binary matching
        idx_pred, idx_target = linear_sum_assignment(C)
        idx_pred = torch.as_tensor(
            idx_pred, dtype=torch.int64, device=pred["cls_prob"].device
        )
        idx_target = torch.as_tensor(
            idx_target, dtype=torch.int64, device=pred["cls_prob"].device
        )

        return idx_pred, idx_target

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_focal: {}".format(self.cost_focal),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def remap_instance_id(instance, pred_instance):
    """按 IoU 将 pred_instance 的id映射为 instance 的id, 输入均为 numpy 数组"""
    instance = instance.astype(np.int32)
    pred_instance = pred_instance.astype(np.int32)
    # 获取唯一的 instance ID
    gt_ids = np.unique(instance)
    pred_ids = np.unique(pred_instance)

    # 计算 IoU 矩阵
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            intersection = np.sum((instance == gt_id) & (pred_instance == pred_id))
            union = np.sum((instance == gt_id) | (pred_instance == pred_id))
            iou_matrix[i, j] = intersection / (union + 1e-8)

    # 使用匈牙利算法进行匹配
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    # 创建 ID 映射字典
    id_mapping = {pred_ids[j]: gt_ids[i] for i, j in zip(gt_indices, pred_indices)}
    # id_mapping[0] = 0  # 背景类映射
    # 重映射预测的 instance ID
    remapped_pred = np.zeros_like(pred_instance)
    for pred_id, gt_id in id_mapping.items():
        remapped_pred[pred_instance == pred_id] = gt_id
    return remapped_pred


if __name__ == "__main__":
    # 创建匹配器实例
    matcher = HungarianMatcher(cost_class=1.0, cost_focal=1.0, cost_dice=1.0)

    # 模拟批次数据
    N = 1000  # 点云中的点数
    Q = 4  # 预测的实例数
    P = 3  # 真实的实例数
    num_classes = 5  # 类别数

    # 创建有意义的预测和目标mask（每个实例占据不同的点集）
    # 创建预测数据
    pred_masks = torch.zeros(Q, N)  # 注意这里维度顺序改为 [Q, N]
    segment_size = N // Q
    for q in range(Q):
        start_idx = q * segment_size
        end_idx = (q + 1) * segment_size if q < Q - 1 else N
        pred_masks[q, start_idx:end_idx] = (
            torch.rand(end_idx - start_idx) * 0.5 + 0.5
        )  # 使值更可能大于0.5

    # 创建目标数据 - 稍微偏移以创建不同的匹配效果
    tgt_masks = torch.zeros(P, N)  # 注意这里维度顺序改为 [P, N]
    segment_size = N // P
    for p in range(P):
        start_idx = p * segment_size
        end_idx = (p + 1) * segment_size if p < P - 1 else N
        tgt_masks[p, start_idx:end_idx] = 1.0  # 目标mask是二进制的

    pred = {
        "cls_prob": torch.randn(Q, num_classes).softmax(dim=-1),  # 类别预测概率
        "masks_heatmap": pred_masks,  # 预测的mask热力图
        "batch": torch.tensor([0, 0, 1, 1]),  # 两个批次的数据
    }

    target = {
        "cls": torch.tensor([1, 2, 3]),  # 目标类别，键名从"labels"改为"cls"
        "masks": tgt_masks,  # 目标mask
        "batch": torch.tensor([0, 0, 1]),  # 对应的批次
    }

    # 运行匹配器 - 移除了多余的batch参数
    idx_pred, idx_target = matcher(pred, target)

    # 打印结果
    print("匹配结果：")
    print(f"预测索引: {idx_pred}")
    print(f"目标索引: {idx_target}")

    # 验证匹配结果
    # 计算所有预测和目标之间的 mask IoU
    def compute_mask_iou(pred_masks, target_masks):
        """计算两组mask之间的IoU矩阵
        Args:
            pred_masks: [Q, N] 预测的mask
            target_masks: [P, N] 目标mask
        Returns:
            iou_matrix: [Q, P] IoU矩阵
        """
        # 将mask转换为布尔值
        pred_masks = pred_masks > 0.5  # [Q, N]
        target_masks = target_masks > 0.5  # [P, N]

        # 计算交集
        intersection = pred_masks.float() @ target_masks.T.float()  # [Q, P]

        # 计算并集
        pred_areas = pred_masks.sum(dim=1).view(-1, 1)  # [Q, 1]
        target_areas = target_masks.sum(dim=1).view(1, -1)  # [1, P]
        union = pred_areas + target_areas - intersection  # [Q, P]

        # 计算IoU
        iou_matrix = intersection / (union + 1e-6)  # 添加小值避免除零
        return iou_matrix

    # 计算mask IoU矩阵
    mask_iou = compute_mask_iou(pred["masks_heatmap"], target["masks"])
    print(f"所有预测和目标之间的 mask IoU: \n{mask_iou}")
