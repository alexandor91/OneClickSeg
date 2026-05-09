"""
group & merge, 参考 SAM3D、pointcept
https://github.com/Pointcept/SegmentAnything3D
pointcept/engines/hooks/evaluator.py InsSegEvaluator
"""

import torch
from torch import nn
import torch.nn.functional as F


class ClicksMerger(nn.Module):
    def __init__(
        self,
        dice_threshold=0.4,
        cls_threshold=0.5,
        random_clicks=False,
        reduction="max",
    ):
        """group & merge, N clicks' pred masks to M instance pred mask; test time, so unbatched!

        Params:
        - dice_threshold: in [0,1), merge those with a similarity bigger than the threshold
        - cls_threshold: in [0,1), merge those with a similarity bigger than the threshold
        - random_clicks: if using random clicks for Tester; if False, use clicks_list to merge instance
        - reduction: how to merge
        """
        super().__init__()
        self.dice_threshold = dice_threshold
        self.cls_threshold = cls_threshold
        self.random_clicks = random_clicks
        self.reduction = reduction
        assert reduction in ["mean", "sum", "max"], "unknown reduction type"

    def associate(self, masks_heatmap, cls_logits, clicks_list):
        """
        - masks_heatmap: N_points x N_clicks, as heatmap
        - cls_logits: N_clicks x num_classes, where 'num_classes' means max semantic label num, unnormalized
        - clicks_list: list of N_clicks

        Return:
        - tag_to_idx: dict(int: 1d Tensor), class_tag -> idx list in clicks_list
        """
        # val set 时clicks可根据instance真值生成, 故有对应的 instance class_tag
        if not self.random_clicks:
            ins_tags = torch.Tensor([click.class_tag for click in clicks_list]).unique()
            tag_to_idx = {i.item(): [] for i in ins_tags}
            for i in range(len(clicks_list)):
                click = clicks_list[i]
                tag_to_idx[click.class_tag].append(i)
            return {
                i.item(): torch.Tensor(tag_to_idx[i]).to(masks_heatmap.device)
                for i in ins_tags
            }
        # test set 时没有真值, 随机clicks, 需要两两判断匹配
        assert cls_logits.shape[0] == masks_heatmap.shape[1]
        tag_to_idx = {}
        clicks_list_idx_out = torch.Tensor([]).to(
            masks_heatmap.device
        )  # 记录已经匹配的 clicks idx
        ins_tag_cnt = 0
        cls_pred = F.layer_norm(cls_logits, [cls_logits.shape[-1]])
        for i in range(cls_logits.shape[0]):
            if i in clicks_list_idx_out:
                continue
            mask = masks_heatmap[:, i]
            dice_coeff = (
                2 * mask @ masks_heatmap / (mask.view(-1, 1) + masks_heatmap).sum(0)
            )  # (N_clicks,)
            cls_similarity = cls_pred[i] @ cls_pred.T  # cos sim
            idx_list = torch.where(
                torch.logical_and(
                    dice_coeff > self.dice_threshold,
                    cls_similarity > self.cls_threshold,
                )
            )[0]
            idx_list = idx_list[~torch.isin(idx_list, clicks_list_idx_out)]
            if len(idx_list) == 0:
                # 正常来讲, i 肯定在 new_idx_list 里, 因为 i not in clicks_list_idx_out, 但保险点, 写了这条
                continue
            clicks_list_idx_out = torch.cat([clicks_list_idx_out, idx_list])
            tag_to_idx[ins_tag_cnt] = idx_list
            ins_tag_cnt += 1
        return tag_to_idx

    def merge(self, tag_to_idx, masks_heatmap, cls_logits):
        # TODO: 区分negtive click
        # TODO: 将相交的区域叠加, 不相交的区域取positive的
        new_masks_heatmap = []
        new_cls_logits = []
        # print(tag_to_idx)
        for ins_tag in tag_to_idx.keys():
            idx_list = tag_to_idx[ins_tag]
            masks = masks_heatmap[:, idx_list].reshape(
                -1, len(idx_list)
            )  # N_points x m_clicks
            clss = cls_logits[idx_list].reshape(
                len(idx_list), -1
            )  # m_clicks x num_classes
            # merge
            if self.reduction == "mean":
                mask_pred = masks.mean(1)  # N_points
                cls_pred = clss.mean(0)  # num_classes
            elif self.reduction == "sum":
                mask_pred = masks.sum(1).sigmoid()
                cls_pred = clss.sum(0)
            elif self.reduction == "max":
                mask_pred = masks.max(1)[0]
                cls_pred = clss.max(0)[0]
            new_masks_heatmap.append(mask_pred.reshape(1, -1))
            new_cls_logits.append(cls_pred.reshape(1, -1))

        return torch.cat(new_masks_heatmap).to(masks_heatmap.device).T, torch.cat(
            new_cls_logits
        ).to(cls_logits.device)

    @torch.no_grad()
    def forward(self, masks_heatmap, cls_logits, clicks_list):
        """
        - masks_heatmap: N_points x N_clicks, as heatmap
        - cls_logits: N_clicks x num_classes, where 'num_classes' means max semantic label num, unnormalized
        - clicks_list: list of N_clicks
        - random_clicks: if using random clicks for Tester

        Return:
        - tag_to_idx: dict(int: 1d Tensor), class_tag -> idx list in clicks_list
        - new_masks_heatmap: N_points x M_pred_instance
        - new_cls_logits: M_pred_instance x num_classes
        """
        tag_to_idx = self.associate(masks_heatmap, cls_logits, clicks_list)
        new_masks_heatmap, new_cls_logits = self.merge(
            tag_to_idx, masks_heatmap, cls_logits
        )
        return tag_to_idx, new_masks_heatmap, new_cls_logits
