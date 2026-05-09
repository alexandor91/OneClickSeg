"""
Instance Segmentation / Interactive Segmentation Engine (Trainer & Tester)

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import wandb
import time
import numpy as np
import torch
import torch.utils.data
from functools import partial

import pointops
from .train import Trainer, TRAINERS
from .test import TesterBase, TESTERS
from .defaults import worker_init_fn
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset_iseg, collate_fn_iseg
from pointcept.utils.logger import get_root_logger
from pointcept.utils.misc import AverageMeter


@TRAINERS.register_module(name="InsSegTrainer")
class InsSegTrainer(Trainer):
    """instance segmentation trainer, collate_fn 换成 instance reid 的"""

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_train_loader(self):
        train_data = build_dataset_iseg(
            self.cfg.data.train, self.cfg.data.ext_valid_assets
        )
        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        # 因为在用DistributedSampler实例化sampler的时候就已经shuffle过数据了
        # 所以在定义dataloader的时候不需要再将shuffle设置为True
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(
                collate_fn_iseg,
                instance_ignore_label=self.cfg.instance_ignore_label,
                mix_prob=self.cfg.mix_prob,
            ),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset_iseg(
                self.cfg.data.val, self.cfg.data.ext_valid_assets
            )
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn_iseg,
                    instance_ignore_label=self.cfg.instance_ignore_label,
                    mix_prob=self.cfg.mix_prob,
                ),
            )
        return val_loader


@TESTERS.register_module()
class InsSegTesterUser(TesterBase):
    """instance segmentation tester modified by user 强制要求 batch_size=1

    Args:
        semantic_ignore: 忽略的语义类别索引
        instance_ignore: 忽略的实例类别索引
        semantic_background: 背景类别索引, 单独计算 ap, 不计入总 ap
    """

    def __init__(
        self,
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background
        self.class_names = self.cfg.data.names
        self.params = {
            "overlaps": self.overlaps,
            "valid_class_tags": [
                i for i in range(len(self.class_names)) if i not in semantic_background
            ],
            "semantic_background": semantic_background,
        }
        self.overlaps = np.append(
            np.arange(0.5, 0.95, 0.05), 0.25
        )  # ins seg 常用 IoU 阈值
        # TODO wandb init define_metric

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()
        scenes = []

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_name = data_dict.pop("name")
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(data_dict)
                segment = data_dict["origin_segment"]
                instance = data_dict["origin_instance"]

                # TODO 改成 inverse
                if "origin_coord" in data_dict.keys():
                    reverse, _ = pointops.knn_query(
                        1,
                        data_dict["coord"].float(),
                        data_dict["offset"].int(),
                        data_dict["origin_coord"].float(),
                        data_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]
                    segment = data_dict["origin_segment"]
                    instance = data_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance
                )

            scenes.append(dict(gt=gt_instances, pred=pred_instance))
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                self.write_scannetpp_results(
                    output_dict["pred_scores"],
                    output_dict["pred_masks"],
                    output_dict["pred_classes"],
                    data_name,
                )

        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def collate_fn(self, batch):
        return batch
