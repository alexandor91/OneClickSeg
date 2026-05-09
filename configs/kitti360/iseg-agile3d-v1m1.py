from pointcept.datasets.preprocessing.kitti360.helpers.labels import trainId2label

KITTI360_CLASS_LABELS = {i: trainId2label[i].name for i in trainId2label.keys()}

_base_ = ["../_base_/insseg_default_runtime.py"]
wandb_project = "AGILE3D"

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
num_worker = 12
mix_prob = 0.8  # mix3d
empty_cache = False
enable_amp = True

num_classes = 20
semantic_ignore = -1
instance_ignore = -1
semantic_background = (0, 1)
matcher_cfg = dict(
    cost_class=1.0, cost_focal=2.0, cost_dice=2.0, instance_ignore=instance_ignore
)
# 数据集相关, max_num_instance <= num_query <=topk_per_scene <= num_query * num_classes
# model settings
model = dict(
    type="Agile3d-v1m1",
    pcd_backbone=dict(
        type="SpUNet-v1m1-fpn",
        in_channels=3,
        # 0 表示不映射到 num_classes
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        # 每个stage的BasicBlock数, BasicBlock不改变分辨率
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    mask_decoder=dict(
        type="Agile3dMaskDecoder",
        transformer_block_cfg=dict(
            type="Mask3dDecoderBlock",
            embedding_dim=128,
            num_heads=8,
            mlp_dim=1024,
            activation="relu",
            attn_drop=0.0,
            layer_drop=0.0,
            norm_before=False,
        ),
        num_classes=num_classes,
        attn_mask_types=["float", "bool", "bool", "bool", "bool"],
        enable_final_block=True,
        num_decoders=3,
        shared_decoder=True,
        mask_num=1,
    ),
    loss=dict(
        type="InsSegLoss",
        cls_loss_cfg=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
                reduction="mean",
                ignore_index=semantic_ignore,
            )
        ],
        mask_loss_cfg=[
            dict(
                type="BinaryCrossEntropyLoss",
                reduction="mean",
                logits=True,
                loss_weight=2.0,
            ),
            dict(
                type="BinaryDiceLoss",
                exponent=1,
                reduction="mean",
                logits=True,
                loss_weight=2.0,
            ),
        ],
    ),
    fused_backbone=False,
    on_segment=False,
    matcher_cfg=matcher_cfg,
    aux=True,
    features_dims=(256, 256, 128, 96, 96),
    num_query=100,
    query_type="sample",
    mask_threshold=0.5,
    topk_per_scene=200,
    semantic_ignore=semantic_ignore,
    instance_ignore=instance_ignore,
    semantic_background=semantic_background,
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="WandbWatch", log="all", log_freq=10),
    dict(
        type="ISegEvaluator",
        semantic_ignore=semantic_ignore,
        instance_ignore=instance_ignore,
        semantic_background=semantic_background,
    ),
    dict(type="CheckpointSaver", save_freq=10),
    dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="InsSegTrainer")
test = dict(type="InsSegTesterUser")

# scheduler settings
evaluate = True
epoch = 200  # 是eval_epoch的整数倍, 通过 cfg.data.train.loop 来实现拼接多个数据循环
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.5,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1.0,
)
# pointcept.utils.optimizer 中会根据 param_dicts 来设置不同的 lr
# param_dicts = [dict(keyword="block", lr=0.00001)]

# dataset settings
dataset_type = "Kitti360Dataset"
data_root = "data/kitti360"

data = dict(
    num_classes=num_classes,
    ignore_index=semantic_ignore,
    names=KITTI360_CLASS_LABELS,
    ext_valid_assets=["sampled_idx_fps_100"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"sampled_idx_fps_100": "sampled_index"}),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SampledIndex2Mask"),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # 这里训练时将 semantic_background 纳入训练？
            dict(
                type="InstanceParser",
                segment_ignore_index=(semantic_ignore, *semantic_background),
                instance_ignore_index=instance_ignore,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "sampled_mask"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"sampled_idx_fps_100": "sampled_index"}),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SampledIndex2Mask"),
            dict(type="SphereCrop", point_max=102400, mode="random"),  # TODO 暂时 crop
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=(semantic_ignore, *semantic_background),
                instance_ignore_index=instance_ignore,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "sampled_mask"),
                feat_keys=("color",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"sampled_idx_fps_100": "sampled_index"}),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",  # 注意这里改成 train type, 不用那么多重复的 fragment
                return_grid_coord=True,
                return_inverse=True,
            ),
            crop=None,
            post_transform=[
                dict(type="SampledIndex2Mask"),
                dict(type="CenterShift", apply_z=False),
                dict(
                    type="InstanceParser",
                    segment_ignore_index=(semantic_ignore, *semantic_background),
                    instance_ignore_index=instance_ignore,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "inverse",
                        "segment",
                        "instance",
                        "sampled_mask",
                    ),
                    feat_keys=("color",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
