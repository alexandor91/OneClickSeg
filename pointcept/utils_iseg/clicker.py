"""
clicker, 参考 EMC-click
https://github.com/feiaxyt/EMC-Click/blob/main/isegm/inference/clicker.py

要求有.set_state(), .update_pred(), .make_init_clicks(), .make_next_clicks()方法
"""

import numpy as np
from copy import deepcopy
import cv2
import torch
import random
import pointops
from ..utils.misc import intersection_and_union_gpu
from ..utils.registry import Registry
from collections.abc import Sequence

CLICKER = Registry("clicker")


# 不能从 pointcept.model.utils 里导入, 会循环引用
def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


@CLICKER.register_module()
class Clicker2D(object):
    def __init__(self, gt_mask, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        """需要将gt_mask按照instance处理成bool"""
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask,
        )
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask,
        )

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return MyClick(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class MyClick:
    def __init__(
        self,
        is_positive,
        coords,
        index=None,
        batch_id=None,
        class_tag=None,
        feat=None,
        pe=None,
    ):
        """
        - coords: 2d时为(y,x), 3d时为(x,y,z)
        - feat: click对应的feature
        - class_tag: click希望分割的class标号
        - batch_id: 在batch中的scene标号
        - index: 保证与 grid_coord 是一致的
        """
        self.is_positive = is_positive
        self.coords = coords
        self.index = index
        self.batch_id = batch_id
        self.class_tag = class_tag
        self.feat = feat
        self.pe = pe

    def copy(self):
        return deepcopy(self)


class Clicks_idx_per_class:
    """为每个 semantic/instance class 维护一个click idx list"""

    def __init__(self):
        self.reset_clicks()

    def add_click(self, click_idx: int, is_positive: bool):
        if is_positive:
            self.pos_clicks_idx_list.append(click_idx)
        else:
            self.neg_clicks_idx_list.append(click_idx)
        self.clicks_idx_list.append(click_idx)

    def get_clicks_idx_list(self):
        return self.clicks_idx_list

    def get_latest_pos_neg_idx(self):
        """return the initial click and newest positive & negative clicks idx; if none, return None
        - 默认initial click为positive"""
        init = None
        pos = None
        neg = None
        n_pos, n_neg = self.cnt()
        if n_pos:
            init = self.pos_clicks_idx_list[0]
            pos = self.pos_clicks_idx_list[-1]
        if n_neg:
            neg = self.neg_clicks_idx_list[-1]
        return init, pos, neg

    def reset_clicks(self):
        self.ious = [0.0]
        self.clicks_idx_list = []
        self.pos_clicks_idx_list = []
        self.neg_clicks_idx_list = []

    def cnt(self):
        return len(self.pos_clicks_idx_list), len(self.neg_clicks_idx_list)

    def __len__(self):
        return len(self.clicks_idx_list)


# class Clicks_per_fragment:
#     '''为每个fragment中的各个类别, 维护一个Clicks_per_class'''
#     def __init__(self, classes):
#         self.classes = classes
#         self.reset_clicks(classes)

#     def add_click(self, click:MyClick, class_tag:int):
#         '''
#         按照class增加click
#         '''
#         self.clicks_list_map[class_tag].add_click(click)

#     def get_clicks(self, class_tag:int):
#         '''
#         按照class获取click
#         '''
#         return self.clicks_list_map[class_tag].get_clicks()

#     def reset_clicks(self, classes):
#         self.clicks_list_map = {i:Clicks_per_class() for i in classes}

#     def get_noc_map(self):
#         '''number of clicks'''
#         return {i:len(self.clicks_list_map[i]) for i in self.classes}


class Clicks_per_fragment:
    """为每个fragment中的各个类别, 维护一个Clicks_per_class"""

    def __init__(self, classes, batch_size):
        # TODO 将classes改成 list[list] 的格式, 而不是现在的 list[tensor], 统一格式, 方便add class
        self.classes = classes  # (bs, N)
        self.batch_size = batch_size
        self.reset_clicks(classes, batch_size)

    def add_click(self, click: MyClick, batch_id: int, class_tag: int):
        """
        按照class增加click
        """
        self.clicks_idx_map_list[batch_id][class_tag].add_click(
            len(self.clicks_list), click.is_positive
        )
        self.clicks_list.append(click)

    def get_clicks_by_class(self, batch_id: int, class_tag: int):
        """
        按照class获取click, 用于考察单次输入多click对同一class的增幅 (unused)
        """
        clicks_idx_list = self.clicks_idx_map_list[batch_id][
            class_tag
        ].get_clicks_idx_list()
        return [self.clicks_list[i] for i in clicks_idx_list]

    def get_clicks(self):
        """all"""
        return self.clicks_list

    def get_latest_clicks(self):
        """
        返回每个instance的initial click 以及最新的 pos & neg clicks 各一个
        - clicks[batch_id][class_tag]['init'] -> MyClick
        """
        clicks = [{i: {} for i in self.classes[j]} for j in range(self.batch_size)]
        for batch_id in range(self.batch_size):
            for class_tag in self.classes[batch_id]:
                init, pos, neg = self.clicks_idx_map_list[batch_id][
                    class_tag
                ].get_latest_pos_neg_idx()
                if init != None:
                    init = self.clicks_list[init]
                    pos = self.clicks_list[pos]  # when only 1 click, pos==click
                if neg != None:
                    neg = self.clicks_list[neg]
                clicks[batch_id][class_tag]["init"] = init
                clicks[batch_id][class_tag]["pos"] = pos
                clicks[batch_id][class_tag]["neg"] = neg
        return clicks

    def get_all_cls_ious(self, batch_id: int):
        """return map: class_tag -> iou_array"""
        return {
            i: np.array(self.clicks_idx_map_list[batch_id][i].ious, float)
            for i in self.classes[batch_id]
        }

    def get_cls_ious(self, batch_id: int, class_tag: int):
        return self.clicks_idx_map_list[batch_id][class_tag].ious

    def update_cls_iou(self, iou, batch_id: int, class_tag: int):
        self.clicks_idx_map_list[batch_id][class_tag].ious.append(iou)

    def reset_clicks(self, classes, batch_size):
        self.clicks_list = []
        self.clicks_idx_map_list = [
            {i: Clicks_idx_per_class() for i in classes[j]} for j in range(batch_size)
        ]


@CLICKER.register_module()
class Clicker3D(object):
    def __init__(
        self,
        semantic_ignore_label=-1,
        instance_ignore_label=-1,
        max_iou=0.95,
        sample_num=200,
    ):
        """
        一个clicker对应一个fragment_list, 需要为同一个batch的
        Argument:
        - semantic_ignore_label: 忽略的 semantic 类别标签, 表示“其它”类
        - instance_ignore_label: 忽略的 instance 类别标签
        - max_iou: 超过这个iou就不需要继续增加clicks了
        - sample_num: is_testset 时有效, 测试集上不是progressive地取click, 而是通过sample fps_num 个点
        """
        self.semantic_ignore_label = semantic_ignore_label
        self.instance_ignore_label = instance_ignore_label
        self.max_iou = max_iou
        # sample_num: is_testset 时有效, 测试集上不是progressive地取click, 而是通过sample
        self.sample_num = sample_num

        # jitter_prob: train mode 下有效, 训练时不直接取中心, 而是在距离中心一定范围内随机取
        self.jitter_prob = 1.0
        # choose_prob: train mode 下有效, 训练时除了随机选的一个instance之外, 其余的都以 1-prob 概率被丢弃
        self.choose_prob = 1.0
        self.cut = 0
        self.on = True

    def set_state(
        self,
        mode,
        fragment_list,
        semantic=None,
        instance=None,
        img_dict=None,
        fusion=None,
    ):
        """
        设置状态, 一个clicker对应一个fragment_list, 均在gpu上
        Argument:
        - fragment_list: voxelize后的data_dict构成的列表, 只读, gpu上的tensor; 若为 train mode, 需要 [input_dict]
        - mode: in ['train','val','test']
        - clicks_from_instance: 是否根据instance gt构建clicks
        - semantic: mask的真值, 1d的gpu上的 torch tensor
            - test: 为未voxelize的点对应的mask; 只有test mode时需要传入; 对 test split, 全为 ignore_label
            - train/val: 直接从fragment中取
        - 若为 train mode, 则 len(fragment_list)=1, len(fragment_list[0]["offset"])>1
        - 若为 test mode, 则 len(fragment_list)>1, len(fragment_list[0]["offset"])=1
        """
        assert mode in ["train", "val", "test"], "unknown mode"
        self.mode = mode
        # 统一train和test的格式, 方便后续处理
        # test mode 在做 grid sample 的时候, 不会sample label, 因为按 index 取就行, 减少做数据增强的显存消耗
        self.is_testset = False
        # self.ignore_label = self.instance_ignore_label if clicks_from_instance else self.semantic_ignore_label
        if self.mode == "test":
            self.semantic = semantic
            self.instance = instance
            self.is_testset = (
                len(self.semantic[self.semantic != self.semantic_ignore_label]) == 0
            )  # 判断是不是 test set 做 tester
        else:
            self.semantic = fragment_list[0]["segment"]
            self.instance = fragment_list[0]["instance"]

        assert isinstance(fragment_list, Sequence) and len(fragment_list) > 0
        self.not_ignore_mask = torch.logical_and(
            self.semantic != self.semantic_ignore_label,
            self.instance != self.instance_ignore_label,
        )
        self.fragment_list = fragment_list
        # self.index = [fragment["index"] if mode == 'test' else torch.arange(0,fragment['offset'][-1], dtype=int) for fragment in fragment_list]
        self.batch_size = len(fragment_list[0]["offset"])
        self.batch_id = offset2batch(fragment_list[0]["offset"])

        self.ins_tag_to_mask = {i: None for i in range(self.batch_size)}
        # batched: 1 bs N fragments or N bs 1 fragment
        self.classes = []
        self.ins_tag_to_sem_label = []
        for b in range(self.batch_size):
            # 对构成batch的每个 scene
            # 对每个instance类别, 维护self.classes, 因为要按照classes取click, 决定哪些class需要继续增加click
            if self.mode == "test":
                batch_semantic = self.semantic
                batch_instance = self.instance
            else:
                batch_mask = self.batch_id == b  # shape as grid_coord
                batch_semantic = self.semantic[batch_mask]
                batch_instance = self.instance[batch_mask]
            # 每个 scene 随机采样部分构建训练样本
            # 构成该batch的scene中所有类别的标号; 对 test_set, 为空array
            unique_cls = torch.unique(batch_instance).int().cpu().numpy()
            unique_cls = unique_cls[unique_cls != self.instance_ignore_label]
            shuffled_idx = list(range(len(unique_cls)))
            random.shuffle(shuffled_idx)
            # shuffled_idx = torch.randperm(len(unique_cls))
            unique_cls = unique_cls[shuffled_idx]
            if len(unique_cls) >= 15 and self.mode == "train":
                unique_cls = unique_cls[0 : len(unique_cls) // 3]
            self.classes.append(unique_cls)
            self.ins_tag_to_sem_label.append(
                {
                    i: batch_semantic[batch_instance == i][0].item()
                    for i in self.classes[-1]
                }
            )
        # 为每个fragment维护一个click_list
        self.clicks = [
            Clicks_per_fragment(self.classes, self.batch_size)
            for i in range(len(fragment_list))
        ]

    # def add_click(self, click):
    # TODO
    #     pass

    # def get_clicks(self, fragment_idx=0):
    #     '''获取classes对应的最后一个click'''
    #     new_clicks = []
    #     for b in range(self.batch_size):
    #         for class_tag in self.classes[b]:
    #             # 对每个class
    #             new_clicks.append(self.clicks[b][fragment_idx].get_clicks(class_tag)[-1])
    #     return new_clicks

    def get_all_clicks(self, fragment_idx=0, clicks_limit=None):
        """获取classes对应的所有clicks"""
        return self.clicks[fragment_idx].get_clicks()[:clicks_limit]

    def get_all_latest_clicks(self, fragment_idx=0):
        """获取classes对应的所有新的clicks"""
        return self.clicks[fragment_idx].get_latest_clicks()

    def generate_fps(self, pcd_dict, fragment_idx=0, fragment_feature=None):
        """
        最远点采样
        - fragment_idx: index of fragment part in fragment_list
        - fragment_feature: (N_points, c)
        TODO distance based fps
        """
        assert self.mode == "test", "fps clicks only used in test"
        p = pcd_dict["grid_coord"].float().cuda()
        o = pcd_dict["offset"].cuda()
        feat = fragment_feature.clone().detach()
        index = torch.arange(p.shape[0])  # 记录 click 在原 pcd 中的 index

        # fps
        for _ in range(4):
            # 4轮fps, 每轮下采样到 1/2, max_point_per_scene==100k, downsample to 6250
            n_o = torch.zeros_like(o).long().to(o.device)
            end_id = 0
            cnt = 0
            for i in range(o.shape[0]):
                # 由offset计算new_offset
                start_id = end_id
                end_id = o[i].item()
                cnt += torch.div((end_id - start_id), 2, rounding_mode="floor")
                n_o[i] = cnt
            sampled_idx = pointops.farthest_point_sampling(p, o, n_o)
            sampled_idx = sampled_idx.long().cpu()
            p = p[sampled_idx]
            feat = feat[sampled_idx]
            index = index[sampled_idx]
            o = n_o.clone().detach()

        # fps with feature distance, distance matrix up to 300 MB  for 6250^2 double type
        # from 3DSSD https://github.com/qiqihaer/3DSSD-pytorch-openPCDet
        n_o = torch.zeros_like(o).long().to(o.device)
        cnt = 0
        for i in range(o.shape[0]):
            # 由offset计算new_offset
            cnt += self.sample_num
            n_o[i] = cnt
        # L2 distance
        feat_norm = feat / ((feat**2).sum(-1) ** 0.5).view(-1, 1)
        dist = 1 - feat_norm @ feat_norm.T
        sampled_idx = (
            pointops.farthest_point_sampling_with_dist(dist, o, n_o).long().cpu()
        )

        # generate clicks
        end_id = 0
        class_tag = (
            self.instance[pcd_dict["index"]].int().cpu().numpy()
        )  # only for test grid sample that returns "index" attribute
        for i in range(n_o.shape[0]):
            # i in batch
            start_id = end_id
            end_id = n_o[i].cpu().item()
            sampled_idx_i = sampled_idx[start_id:end_id]
            coords_i = p[sampled_idx_i]
            index_i = index[sampled_idx_i]
            class_tag_i = class_tag[index_i]
            for j in range(sampled_idx_i.shape[0]):
                if index_i[j] >= pcd_dict["grid_coord"].shape[0] - 10:
                    # 奇怪的bug, scene0696_01, 第9个fragment的size跟其它的不一样, 为了节省时间对同一scene不同fragment采用同样的clicks, 所以得删去超出范围的
                    continue
                click = MyClick(
                    True,
                    coords_i[j],
                    batch_id=i,
                    class_tag=class_tag_i[j],
                    index=index_i[j],
                )
                self.clicks[fragment_idx].add_click(
                    click, batch_id=i, class_tag=0
                )  # 不管 class_tag, 一视同仁
        return self.get_all_clicks(fragment_idx=fragment_idx)

    def make_init_clicks(self, fragment_idx=0, fragment_feature=None):
        """
        instance的中点
        - fragment_idx: 0 if not test mode
        - fragment_feature: None if not test mode
        """
        fragment = self.fragment_list[fragment_idx]
        if self.is_testset:
            # 对测试集, 没有真值, 随机采点策略
            return self.generate_fps(
                fragment, fragment_idx=fragment_idx, fragment_feature=fragment_feature
            )
        for b in range(self.batch_size):
            select_cls = random.randint(0, len(self.classes[b]) - 1)
            # 限制 instance 的个数: 训练时除了随机选的一个instance之外, 其余的都以 1-prob 概率被丢弃; 非训练时 prob==1
            if_select = random.choices(
                [False, True],
                [1 - self.choose_prob, self.choose_prob],
                k=len(self.classes[b]),
            )
            if_select[select_cls] = True
            for class_tag in self.classes[b][if_select]:
                # for class_tag in self.classes[b]:
                # 对每个class
                if self.mode == "test":
                    # test_mode的voxelize才会返回index, 因为train mode不需要index, 直接返回indexed instance
                    mask = self.instance[fragment["index"]] == class_tag
                else:
                    # 保证mask与grid_coord.shape[0]一致
                    batch_mask = self.batch_id == b
                    mask = self.instance == class_tag
                    mask = torch.logical_and(mask, batch_mask)
                grid_coord = fragment["grid_coord"][
                    mask.bool()
                ]  # 该scene中对应class_tag的voxel
                index = torch.arange(0, fragment["grid_coord"].shape[0])[mask.bool()]
                weight_center = grid_coord.float().mean(0).long()  # 重心
                distance = ((grid_coord - weight_center) ** 2).sum(1)
                if self.mode != "train":
                    # 找voxel集中离重心近的
                    select_idx = distance.argmin()
                    point = grid_coord[select_idx]
                    index = index[select_idx]
                else:
                    # 在中心附近的一定范围内随机选取
                    # radius = distance.float().mean()         # 半径均值
                    max_distance = max(
                        self.jitter_prob * distance.max(), distance.min()
                    )
                    # print(grid_coord.shape, distance.shape, radius, class_tag)
                    mask = distance <= max_distance
                    grid_coord = grid_coord[mask]
                    index = index[mask]
                    select_idx = torch.randint(grid_coord.shape[0], (1,))[0]
                    point = grid_coord[select_idx]
                    index = index[select_idx]
                # index 变成0dim的tensor
                click = MyClick(
                    True, point, batch_id=b, class_tag=class_tag, index=index
                )
                self.clicks[fragment_idx].add_click(
                    click, batch_id=b, class_tag=class_tag
                )
        return self.get_all_latest_clicks(fragment_idx=fragment_idx)

    def make_next_clicks(self, fragment_idx=0):
        """
        迭代地增加click
        - jitter_prob: train mode 下有效, 训练时不直接取中心, 而是在距离中心一定范围内随机取
        """
        assert self.instance is not None
        fragment = self.fragment_list[fragment_idx]
        self.on = False  # 判断本次 make_next_click 是否增加了click
        for b in range(self.batch_size):
            select_cls = random.randint(0, len(self.classes[b]) - 1)
            # 限制 instance 的个数: 训练时除了随机选的一个instance之外, 其余的都以 1-prob 概率被丢弃; 非训练时 prob==1
            if_select = random.choices(
                [False, True],
                [1 - self.choose_prob, self.choose_prob],
                k=len(self.classes[b]),
            )
            if_select[select_cls] = True
            for class_tag in self.classes[b][if_select]:
                # for class_tag in self.classes[b]:
                # 对每个class
                if (
                    self.clicks[fragment_idx].get_cls_ious(b, class_tag)[-1]
                    >= self.max_iou
                ):
                    # 如果iou达标, 则不需要继续增加click
                    continue
                if self.mode == "test":
                    # test mode 时 instance 到 grid_coord 需要 index, 因为instance对应的是未voxelize的
                    # 需要产生的clicks对应于voxel
                    gt_mask = self.instance[fragment["index"]] == class_tag
                    pred_mask = self.class_tag_to_mask[class_tag][fragment["index"]]
                    not_ignore_mask = self.not_ignore_mask[fragment["index"]]
                else:
                    # train mode 时 instance 和 pred 形状都与 grid_coord.shape[0] 一致
                    batch_mask = self.batch_id == b
                    gt_mask = self.instance == class_tag
                    gt_mask = torch.logical_and(gt_mask, batch_mask)
                    pred_mask = self.class_tag_to_mask[b][class_tag]
                    not_ignore_mask = self.not_ignore_mask

                tp_mask = torch.logical_and(gt_mask, pred_mask)
                fn_mask = torch.logical_and(gt_mask, torch.logical_not(pred_mask))
                # 逐个 class_tag 判断是否需要增加 click
                iou = tp_mask.sum() / torch.logical_or(gt_mask, pred_mask).sum()
                if iou > self.max_iou and self.mode == "test":
                    continue
                # if 1- (fn_mask.sum() / tp_mask.sum()) > self.max_recall:
                #     continue
                fp_mask = torch.logical_and(
                    torch.logical_and(torch.logical_not(gt_mask), pred_mask),
                    not_ignore_mask,
                )
                # if 1- fn_mask.sum() / (gt_mask * pred_mask).sum() > self.max_precision:
                #     continue
                # 判断要取正还是取负, 按面积
                if fn_mask.sum() > fp_mask.sum():
                    is_positive = True
                    mask = fn_mask
                else:
                    is_positive = False
                    mask = fp_mask
                # 取重心
                grid_coord = fragment["grid_coord"][mask]
                index = torch.arange(0, fragment["grid_coord"].shape[0])[mask]
                if grid_coord.shape[0] == 0:
                    # 训练阶段mask后可能是0, 因为fn_mask可能为0
                    continue
                self.on = True
                weight_center = grid_coord.float().mean(0).long()  # 重心
                distance = ((grid_coord - weight_center) ** 2).sum(1)  # 距离
                if self.mode != "train":
                    # 找voxel集中离重心近的
                    select_idx = distance.argmin()
                    point = grid_coord[select_idx]
                    index = index[select_idx]
                else:
                    # 在中心附近的一定范围内随机选取
                    # radius = distance.float().mean()         # 半径均值
                    max_distance = max(
                        self.jitter_prob * distance.max(), distance.min()
                    )
                    mask = distance <= max_distance
                    grid_coord = grid_coord[mask]
                    index = index[mask]
                    select_idx = torch.randint(grid_coord.shape[0], (1,))[0]
                    point = grid_coord[select_idx]
                    index = index[select_idx]
                click = MyClick(
                    is_positive, point, batch_id=b, class_tag=class_tag, index=index
                )
                self.clicks[fragment_idx].add_click(
                    click, batch_id=b, class_tag=class_tag
                )
        return self.get_all_latest_clicks(fragment_idx=fragment_idx)

    def update_ious(self, class_tag_to_iou, batch_id=0):
        """
        更新 instance masks 的 pred & tgt iou, 对各个fragment都更新同样的
        - class_tag_to_iou: class_tag -> iou_arr
        """
        for fragment_idx in range(len(self.clicks)):
            for class_tag in class_tag_to_iou.keys():
                self.clicks[fragment_idx].update_cls_iou(
                    class_tag_to_iou[class_tag].item(), batch_id, class_tag
                )

    def update_pred(self, class_tag_to_mask):
        """
        更新 instance masks 对应的 pred mask, 对各个fragment都更新同样的
        - class_tag_to_mask: class_tag -> 1d mask
        - if not test mode, self.class_tag_to_mask[batch_id][class_tag] -> 1d mask
        """
        self.class_tag_to_mask = class_tag_to_mask

    def get_ious(self, fragment_idx=0):
        """返回iou per ins_tag, 仅 test_mode 使用, 因此没做 batched 处理"""
        return self.clicks[fragment_idx].get_all_cls_ious(0)
