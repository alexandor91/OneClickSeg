"""
Interactive Segmentation Hook

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist

from .default import HookBase
from .builder import HOOKS
from pointcept.utils import comm
from pointcept.utils_iseg.ins_seg import associate_matched_ins, evaluate_matched_ins


@HOOKS.register_module()
class ISegEvaluator(HookBase):
    """验证集上, 测试 instance segmentation metric\n
    相当于把 .evaluator.py 中的 InsSegEvaluator 拆分了可复用的部分出去\n
    需要自行在 model inference 中指定 matched_idx,
    因此也可以用来验证 center point query 的 AP@1
    """

    def __init__(
        self, semantic_ignore=-1, instance_ignore=-1, semantic_background=(0, 1)
    ):
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore  # 没用到, mask 在 get_target 中构建成了 [N_ins, N_point] 的形状
        self.semantic_background = semantic_background
        self.class_names = None  # update in before train
        self.overlaps = np.append(
            np.arange(0.5, 0.95, 0.05), 0.25
        )  # ins seg 常用 IoU 阈值

    def before_train(self):
        self.class_names = self.trainer.cfg.data.names
        self.params = {
            "overlaps": self.overlaps,
            "valid_class_tags": [
                i
                for i in range(len(self.class_names))
                if i not in (self.semantic_ignore,)
            ],
        }
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        # TODO 计算一下各个类别的 metric
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            # turn to gpu
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            pred = output_dict["pred"]
            target = output_dict["target"]
            masks_iou = output_dict["masks_iou"]
            cls_precision = output_dict["cls_precision"]
            # void = input_dict["segment"] == self.semantic_ignore
            # 不映射回 gridsample 前的数据, 直接用 voxelization 后的
            if comm.get_world_size() > 1:
                dist.all_reduce(masks_iou), dist.all_reduce(cls_precision)
                # 在all_reduce后除以world_size来计算平均值
                masks_iou = masks_iou / comm.get_world_size()
                cls_precision = cls_precision / comm.get_world_size()
            masks_iou, cls_precision = (
                masks_iou.cpu().numpy(),
                cls_precision.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_masks_iou", masks_iou)
            self.trainer.storage.put_scalar("val_cls_precision", cls_precision)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            del pred, target
            torch.cuda.empty_cache()

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Val: [{iter}/{max_iter}] "
                "Loss {loss:.4f} "
                "mIoU {masks_iou:.4f} "
                "cls_precision {cls_precision:.4f}".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.val_loader),
                    loss=loss.item(),
                    masks_iou=masks_iou.item(),
                    cls_precision=cls_precision.item(),
                )
            )

        # 每个 GPU/进程会收集自己处理的验证数据，存储在 scenes 列表中
        # 然后通过 comm.gather() 函数将所有进程的 scenes 收集到主进程（rank 0）
        # 这会产生一个嵌套列表 scenes_sync，其结构为 [[进程0的scenes], [进程1的scenes], ...]
        comm.synchronize()
        # scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        # ap_scores, _ = evaluate_matched_ins(self.params, scenes)

        # all_ap = ap_scores["all_ap"]
        # all_ap_50 = ap_scores["all_ap_50%"]
        # all_ap_25 = ap_scores["all_ap_25%"]
        # self.trainer.logger.info(
        #     "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
        #         all_ap, all_ap_50, all_ap_25
        #     )
        # )
        # for class_tag in self.params['valid_class_tags']:
        #     ap = ap_scores["classes"][class_tag]["ap"]
        #     ap_50 = ap_scores["classes"][class_tag]["ap50%"]
        #     ap_25 = ap_scores["classes"][class_tag]["ap25%"]
        #     self.trainer.logger.info(
        #         "Class_{idx}-{name} Result: mAP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
        #             idx=class_tag, name=self.class_names[class_tag], AP=ap, AP50=ap_50, AP25=ap_25
        #         )
        #     )
        # current_epoch = self.trainer.epoch + 1
        # if self.trainer.writer is not None:
        #     self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
        #     self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
        #     self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
        #     self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        # self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        # self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        # self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver

        loss_avg = self.trainer.storage.history("val_loss").avg
        masks_iou = self.trainer.storage.history("val_masks_iou").avg
        cls_precision = self.trainer.storage.history("val_cls_precision").avg
        self.trainer.logger.info(
            f"Val: Loss {loss_avg:.4f} masks_iou {masks_iou:.4f} cls_precision {cls_precision:.4f}"
        )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/masks_iou", masks_iou, current_epoch)
            self.trainer.writer.add_scalar(
                "val/cls_precision", cls_precision, current_epoch
            )
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/masks_iou": masks_iou,
                        "val/cls_precision": cls_precision,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = masks_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "masks_iou"  # save for saver

    def after_train(self):
        # self.trainer.logger.info("Best {}: {:.4f}".format(
        #     "AP50", self.trainer.best_metric_value))
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("masks_iou", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class WandbWatch(HookBase):
    """wandb watch model
    TODO 可以只 watch 部分参数, 参考 utils/optimizer.py build_optimizer 的 param_dicts
    """

    def __init__(self, log="all", log_freq=10):
        self.log = log
        self.log_freq = log_freq

    def before_train(self):
        if self.trainer.cfg.enable_wandb and comm.is_main_process():
            # 只在主进程中执行 watch
            wandb.watch(self.trainer.model, log=self.log, log_freq=self.log_freq)
