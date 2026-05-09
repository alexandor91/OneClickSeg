"""
Instance Segmentation Utils

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
from addict import Dict


def unique_id(instance, ignore=-1):
    """
    - instance: [N_point,]
    - ignore: ignore instance id
    - return: [N_instance,]
    """
    return torch.unique(instance[instance != ignore]).to(instance.device).int()


def id_to_mask(instance, ins_id):
    """
    - instance: [N_point,]
    - ins_id: [N_instance,]
    - return: [N_instance, N_point], binary mask
    """
    # 向量化实现
    n_point = instance.shape[0]
    n_instance = ins_id.shape[0]
    # 将 instance 扩展为 [N_instance, N_point]
    instance_expanded = instance.unsqueeze(0).expand(n_instance, n_point)
    # 将 ins_id 扩展为 [N_instance, N_point]
    ins_id_expanded = ins_id.unsqueeze(1).expand(n_instance, n_point)
    # 直接比较生成掩码
    mask = (instance_expanded == ins_id_expanded).float()
    return mask


def mask_to_id(mask):
    """panoptic mask to instance id
    - mask: [N_instance, N_point], binary mask
    - return: [N_point,], instance id
    """
    return torch.argmax(mask, dim=0)


def id_to_other_id(other, instance, ins_id):
    """通过 idx 找出 instance id 对应的 other id
    - other: [N_point,]; such as semantic, batch
    - instance: [N_point,]
    - ins_id: [N_instance,]\n
    Returns:
    - other_id [N_instance,], corresponding to ins_id with same idx
    """
    other_id = torch.zeros_like(ins_id).to(ins_id.device).int()
    for i, id_val in enumerate(ins_id):
        other_id[i] = other[instance == id_val][0]
    return other_id


def get_pred(pred, query):
    """
    - pred: dict, 预测结果
        - 'cls_logits': [N_query, num_classes]
    - query: query dict
        - 'batch': [N_query,]
    Returns:
    - pred dict
        - cls_logits: [N_query, num_classes]
        - cls_prob: [N_query, num_classes], for matcher
        - cls: [N_query,], for eval & test
        - batch: [N_query,]
        - other attributes in input dict, especially when not training, return 'masks' 'scores'
    """
    cls_prob = pred["cls_logits"].softmax(-1)
    cls = cls_prob.argmax(dim=1)
    return Dict(cls_prob=cls_prob, cls=cls, batch=query["batch"], **pred)


def get_target(scene, ignore):
    """reconstruct the ground truth
    - scene: Scene dict
        - semantic: [N_point,]
        - instance: [N_point,]
        - batch_list[-1]: [N_point,]
    - ignore: ignore instance id

    Returns:
    - target dict
        - ins_id: [N_instance,], instance id
        - masks: [N_instance, N_point], binary mask
        - cls: [N_instance,], semantic id
        - batch: [N_instance,], batch id
        - is_testset: bool, if testing on test set
    """
    semantic, instance = scene.semantic, scene.instance
    batch = scene.batch_list[-1]
    ins_id = unique_id(instance, ignore)
    sem_id = torch.zeros_like(ins_id).to(ins_id.device).int()
    batch_id = torch.zeros_like(ins_id).to(ins_id.device).int()
    masks_target = id_to_mask(instance, ins_id)
    is_testset = len(ins_id) == 0

    for i in range(len(ins_id)):
        ins_filter = instance == ins_id[i]
        sem_id[i] = semantic[ins_filter][0]
        batch_id[i] = batch[ins_filter][0]
    # TODO 统一名称: ins, sem 代替 cls
    return Dict(
        ins_id=ins_id,
        masks=masks_target,
        cls=sem_id,
        batch=batch_id,
        is_testset=is_testset,
    )


def softgroup_post_process(pred, topk_per_scene: int):
    """
    test time post process before match, soft voting with high cls_logits, 来自 SoftGroup arXiv:2203.01509
    modified from [Mask3D](https://github.com/JonasSchult/Mask3D/blob/main/trainer/trainer.py#L577)
    - pred: dict, 预测结果
        - masks: (N_query, N_point), float tensor binary mask
        - masks_logits: (N_query, N_point) as logits
        - cls_logits: (N_query, num_classes)
        - other attributes with (N_query, ...)
    - topk_per_scene: 每个场景预测的 instance 数上限

    Returns: updated pred dict
    - masks: (N_pred_instance, N_point), float tensor binary mask
    - masks_logits: (N_pred_instance, N_point)
    - cls: (N_pred_instance,)
    - cls_prob: (N_pred_instance, num_classes)
    - cls_logits: (N_pred_instance, num_classes)
    - scores: (N_pred_instance,)
    - other attributes with (N_pred_instance, ...)
    """
    cls_logits = pred["cls_logits"]
    num_queries, num_classes = cls_logits.shape[-2:]

    # 减小分类错误带来的影响: 一个query可能生成多个instance, 如果它的 cls_pred 值互相之间比较接近
    labels = (
        torch.arange(num_classes, device=cls_logits.device)
        .unsqueeze(0)
        .repeat(num_queries, 1)
        .flatten(0, 1)
    )
    if topk_per_scene < num_queries * num_classes:
        cls_scores_per_query, topk_indices = cls_logits.flatten(0, 1).topk(
            topk_per_scene, sorted=True
        )
    else:
        cls_scores_per_query, topk_indices = cls_logits.flatten(0, 1).topk(
            num_queries * num_classes, sorted=True
        )

    labels_per_query = labels[topk_indices]  # [N_pred_instance,]
    # 从按 N_query*num_classes 的索引, 恢复出按 N_query 的索引
    topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
    for key in pred.keys():
        pred[key] = pred[key][topk_indices]  # [N_pred_instance, ...]

    # 重构 cls_scores_per_query 和 labels_per_query 为 cls_logits
    # [N_pred_instance, num_classes], 非预测label的位置全0
    cls_logits = torch.zeros((len(topk_indices), num_classes), device=cls_logits.device)
    cls_logits[torch.arange(len(topk_indices)), labels_per_query] = cls_scores_per_query
    pred["cls"] = labels_per_query
    pred["cls_prob"] = cls_logits.softmax(-1)
    pred["cls_logits"] = cls_logits

    mask_scores_per_query = (pred["masks_logits"].sigmoid() * pred["masks"]).sum(1) / (
        pred["masks"].sum(1) + 1e-6
    )
    pred["scores"] = cls_scores_per_query * mask_scores_per_query  # scores per query
    return pred


@torch.no_grad()
def associate_matched_ins(pred, target, void):
    """
    整理, 去掉大的数据 (N_point值很大), 保留计算 AP 所用的小的数据
    - pred: dict, 预测结果
        - 'masks': [N_pred, N_point], binary mask
        - 'cls': [N_pred,], semantic id
        - 'scores': [N_pred,], pred confidence score
        - 'matched_idx': [N_matched,], matched pred idx
    - target: dict, 目标结果
        - 'masks': [N_target, N_point], binary mask
        - 'cls': [N_target,], semantic id
        - 'matched_idx': [N_matched,], matched pred idx
    - void: [N_point,], void mask\n
    Returns:
    - pred: dict, 预测结果
        - 'vert_count': [N_pred,], pred vert count
        - 'cls': [N_pred,], pred semantic id
        - 'scores': [N_pred,], pred confidence score
        - 'void_intersection': [N_pred,], pred void intersection
    - target: dict, 目标结果
        - 'vert_count': [N_target,], target vert count
        - 'cls': [N_target,], target semantic id
        - 'idx_to_pred_idx': [N_matched,], matched target idx
    - iou: [N_pred, N_target], IoU of pred and target
    """
    # TODO 删除大的数据, 为什么没有 masks_logits?
    idx_pred, idx_target = pred.pop("matched_idx"), target.pop("matched_idx")
    pred_masks, target_masks = pred.pop("masks"), target.pop("masks")
    pred_new, target_new = {}, {}
    pred_new["cls"], target_new["cls"] = pred.pop("cls"), target.pop("cls")
    pred_new["scores"] = pred.pop("scores")
    # 计算面积
    pred_new["vert_count"] = pred_masks.sum(1)
    target_new["vert_count"] = target_masks.sum(1)
    # 建立索引
    target_new["idx_to_pred_idx"] = {
        idx_target[i].item(): idx_pred[i].item() for i in range(len(idx_pred))
    }
    # pred_masks 和 ignore_mask 的交集
    pred_new["void_intersection"] = (pred_masks * void).sum(1)
    # 计算 IoU
    intersection = pred_masks @ target_masks.T  # [N_pred, N_target]
    union = pred_new["vert_count"].view(-1, 1) + target_new["vert_count"].view(
        1, -1
    )  # [N_pred, N_target]
    union = union - intersection
    iou = intersection / union
    return pred_new, target_new, iou


@torch.no_grad()
def evaluate_matched_ins(params, scenes):
    """
    modified from scannet benchmark's evaluation
    其中 pred 和 target 内部不能有自重复, 应为 unselected 的结果, 即与 select_matched_pred_target 冲突
    - params: dict, ins seg 测试相关参数
        - 'overlaps': list, 评估AP的IoU阈值
        - 'valid_class_tags': list, 评估AP的语义标签
    - scenes: list, 每个场景的预测和目标结果
        - pred: dict, 预测结果
            - 'vert_count': [N_pred,], pred vert count
            - 'cls': [N_pred,], pred semantic class
            - 'scores': [N_pred,], pred confidence score
            - 'void_intersection': [N_pred,], pred void intersection
        - target: dict, 目标结果
            - 'vert_count': [N_target,], target vert count
            - 'cls': [N_target,], target semantic class
            - 'idx_to_pred_idx': [N_matched,], matched target idx
        - iou: [N_pred, N_target], IoU of pred and target\n
    Returns:
    - ap_table: np.ndarray,
    - tgt_idx_to_matched_iou: dict, 每个target idx对应的matched iou
    """
    # TODO 改到 gpu 上进行
    # https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
    # https://github.com/Pointcept/Pointcept/blob/main/pointcept/engines/hooks/evaluator.py
    # TODO: 这里默认 semantic class 为 0 ~ num_classes - 1
    overlaps = params["overlaps"]
    valid_class_tags = params["valid_class_tags"]
    # results: [num_valid_classes, num_overlaps], 每个类别和IoU阈值的AP值
    ap_table = np.zeros((len(valid_class_tags), len(overlaps)), float)
    tgt_idx_to_matched_iou = {}

    for oi, overlap_th in enumerate(overlaps):
        for li, class_tag in enumerate(valid_class_tags):
            y_true = np.empty(0)
            y_score = np.empty(0)
            hard_false_negatives = 0
            has_gt = False
            has_pred = False
            # 遍历每个 scene, 收集 class_tag 的按 overlap_th 判断的测准与否的数目
            for scene in scenes:
                pred = scene["pred"]
                target = scene["target"]
                classi_pred_idx = torch.where(pred["cls"] == class_tag)[0]
                classi_tgt_idx = torch.where(target["cls"] == class_tag)[0]

                if len(classi_pred_idx):
                    has_pred = True
                if len(classi_tgt_idx):
                    has_gt = True
                cur_true = np.ones(len(classi_tgt_idx))
                cur_score = np.ones(len(classi_tgt_idx)) * (-float("inf"))
                cur_match = np.zeros(len(classi_tgt_idx), dtype=bool)
                # collect matches
                for gti, gt in enumerate(classi_tgt_idx):
                    if gt.item() in target["idx_to_pred_idx"].keys():
                        pr = target["idx_to_pred_idx"][gt.item()]
                        # already assigned by matcher
                        ious = scene["iou"][:, gt]
                        iou = ious[pr]
                        tgt_idx_to_matched_iou[gt] = iou
                        # print(f'gt {gt}/{type(gt)}, pr {pr}/{type(pr)}, iou {iou}, max iou {ious.max()}')
                        if iou > overlap_th and torch.isin(pr, classi_pred_idx):
                            cur_match[gti] = True
                            cur_score[gti] = pred["scores"][pr]
                            # append others as false positive
                            prs = torch.where(ious > overlap_th)[0]
                            prs = prs[torch.isin(prs, classi_pred_idx)]
                            prs = prs[prs != pr]
                            for pr in prs:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(
                                    cur_score, pred["scores"][pr].cpu()
                                )
                                cur_match = np.append(cur_match, True)
                        else:
                            # under thershold or not same cls, so this gt keep unmatched
                            hard_false_negatives += 1
                # remove non-matched ground truth instances
                cur_true = cur_true[cur_match]
                cur_score = cur_score[cur_match]
                # collect non-matched predictions as false positive
                for pr in classi_pred_idx:
                    ious = scene["iou"][pr, :]
                    gts = torch.where(ious > overlap_th)[0]
                    gts = gts[torch.isin(gts, classi_tgt_idx)]
                    found_gt = len(gts) > 0
                    if not found_gt:
                        num_ignore = pred["void_intersection"][pr]
                        proportion_ignore = float(num_ignore) / pred["vert_count"][pr]
                        # if not ignored append false positive
                        if proportion_ignore <= overlap_th:
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, pred["scores"][pr].cpu())
                # append to overall results
                y_true = np.append(y_true, cur_true)
                y_score = np.append(y_score, cur_score)

            # compute average precision
            if has_gt and has_pred:
                # compute precision recall curve first
                # sorting and cumsum
                # print(f'class tag {class_tag}')
                # print(f'tgt idx {classi_tgt_idx}')
                # print(f'y score {y_score}')
                score_arg_sort = np.argsort(y_score)
                y_score_sorted = y_score[score_arg_sort]
                y_true_sorted = y_true[score_arg_sort]
                y_true_sorted_cumsum = np.cumsum(y_true_sorted)
                # unique thresholds
                (thresholds, unique_indices) = np.unique(
                    y_score_sorted, return_index=True
                )
                num_prec_recall = len(unique_indices) + 1
                # prepare precision recall
                num_examples = len(y_score_sorted)
                # https://github.com/ScanNet/ScanNet/pull/26
                # all predictions are non-matched but also all of them are ignored and not counted as FP
                # y_true_sorted_cumsum is empty
                # num_true_examples = y_true_sorted_cumsum[-1]
                num_true_examples = (
                    y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                )
                precision = np.zeros(num_prec_recall)
                recall = np.zeros(num_prec_recall)
                # deal with the first point
                y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                # deal with remaining
                for idx_res, idx_scores in enumerate(unique_indices):
                    cumsum = y_true_sorted_cumsum[idx_scores - 1]
                    tp = num_true_examples - cumsum
                    fp = num_examples - idx_scores - tp
                    fn = cumsum + hard_false_negatives
                    p = float(tp) / (tp + fp)
                    r = float(tp) / (tp + fn)
                    precision[idx_res] = p
                    recall[idx_res] = r
                # first point in curve is artificial
                precision[-1] = 1.0
                recall[-1] = 0.0
                # print(precision, recall)
                # compute average of precision-recall curve
                recall_for_conv = np.copy(recall)
                recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                recall_for_conv = np.append(recall_for_conv, 0.0)
                stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                # integrate is now simply a dot product
                ap_current = np.dot(precision, stepWidths)
            elif has_gt:
                ap_current = 0.0
            else:
                ap_current = float("nan")
            ap_table[li, oi] = ap_current

    # compute mAP, AP50, AP25
    o50 = np.where(np.isclose(overlaps, 0.5))
    o25 = np.where(np.isclose(overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(overlaps, 0.25)))
    ap_scores = dict()
    ap_scores["all_ap"] = np.nanmean(ap_table[:, oAllBut25])
    ap_scores["all_ap_50%"] = np.nanmean(ap_table[:, o50])
    ap_scores["all_ap_25%"] = np.nanmean(ap_table[:, o25])
    ap_scores["classes"] = {}
    for li, class_tag in enumerate(valid_class_tags):
        ap_scores["classes"][class_tag] = {}
        ap_scores["classes"][class_tag]["ap"] = np.nanmean(ap_table[li, oAllBut25])
        ap_scores["classes"][class_tag]["ap50%"] = np.nanmean(ap_table[li, o50])
        ap_scores["classes"][class_tag]["ap25%"] = np.nanmean(ap_table[li, o25])
    return ap_scores, tgt_idx_to_matched_iou


if __name__ == "__main__":
    # 测试get_target函数
    n_point = 100
    semantic = torch.full((n_point,), -1)  # 所有语义标签为-1
    instance = torch.full((n_point,), -1)  # 所有实例标签为-1
    batch = torch.zeros(n_point)  # 所有批次ID为0
    ignore = -1  # 忽略值为-1
    target, is_testset = get_target(semantic, instance, batch, ignore)
    print("ins_id:", unique_id(instance, ignore))
    print("target.masks.shape:", target.masks.shape)
    print("target.cls.shape:", target.cls.shape)
    print("target.batch.shape:", target.batch.shape)
    print("is_testset:", is_testset)
