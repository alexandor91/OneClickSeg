# 魔改pointcept框架，增加interactive segmentation的支持
based on 1dd24a07e3533c414539e0d3b918cf793a30d3ff commit

- 在`pointcept/datasets/scannet.py`中增加DATASETS：RGBD image的dataloader（注意统一数据格式，示例为`pointcept/datasets/defaults.py`），以及image和点云集合的datalodaer
    - 预处理那里还需要split成train、val、test
    - 还没写完
- 在`pointcept/models/default.py`中增加MODELS：interactive的base MODELS，支持build两种模态的模型
- 在`pointcept/engines/test.py`中增加TEST：新建测试pipeline，直接在这里面写evaluation metric
    - 需要增加 interactive segmentation 的选点策略、评估指标
    - 考虑增加hiou调和平均


TODO：
- `pointcept/engines/hooks`底下的文件是干啥的？好像在cfg里需要相应更改semantic/instance？
    - 好像是只在`pointcept/engines/train.py`定义的训练pipeline中出现，测试pipeline只需要更改`pointcept/engines/test.py`
- meta transformer


DONE:
- `libs/pointops/`: 添加 fps with distance
    - `src/sampling/`, 源码与wrapper
    - `functions/`, python api
- `libs/scannet_segmentator`
    - https://github.com/yzj2019/scannet_segmentator
- pointcept.datasets.scannet.ScanNetDataset
    - get_idx_by_name, get_data_by_name
- pointcept.datasets.scannet.ScanNetImageDataset
- pointcept.datasets.scannet.ScanNetFusionDataset
- pointcept.datasets.utils.num_to_natural
- pointcept.datasets.utils.fusion_collate_fn
- pointcept.models.losses.misc.BinaryCrossEntropyLoss
- pointcept.models.losses.misc.BinaryDiceLoss
- pointcept.models.default.DefaultInteractiveSegmentor
    - 记得修改 `pointcept.models.__init__.py`
- pointcept.models.default.DefaultSegmentorPretrained
    - 在__init__()中加载ckpt
- pointcept.models.default.DefaultSegmentorFPN
    - 返回 FPN
- pointcept.models.sparse_unet.spconv_unet_v1m1_base.SpUNetBase
    - out_fpn, 返回特征金字塔(FPN)


TODO 待修改
- pointcept.utils.clicker
- pointcept.tools.isegvis
- pointcept.engines.test.InteractiveSemSegTester
- pointcept.models.interactive
    - pointcept.models.interactive.spunet_sam:
- `configs/_base_/interseg_default_runtime.py`
- `configs/scannet200/intersemseg-spunet-v1m1-0-base.py`

- 构建存储库时，记得add sub module 


