# 数据准备&预处理

> Download dataset with [opendatalab](https://opendatalab.com/) api.
> All path must be absolute path

## 1. Scannetv2

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Run preprocessing code for raw ScanNet at `${CODEBASE_DIR}` as follows:
```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```
Link processed dataset to codebase:
```bash
mkdir -p ${CODEBASE_DIR}/data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

## 2. S3DIS

Download [Pointcept preprocessed S3DIS dataset](https://huggingface.co/datasets/Pointcept/s3dis-compressed) and unzip.

Link dataset to the codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```

## 3. SemanticKITTI
Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset. Link dataset to codebase.

```bash
# SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
# |- SEMANTIC_KITTI_DIR
#   |- dataset
#     |- sequences
#       |- 00
#       |- 01
#       |- ...
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```

## 4. [Unofficial] KITTI-360

Download [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) dataset. 

Preprocess KITTI-360 dataset.
```bash
# KITTI_360_DIR: the directory of downloaded KITTI-360 dataset.
# |- KITTI_360_DIR
#   |- data_3d_semantics
# PROCESSED_KITTI_360_DIR: the directory of the processed KITTI-360 dataset (output dir).
python pointcept/datasets/preprocessing/kitti360/preprocess_kitti360.py --dataset_root ${KITTI_360_DIR} --output_root ${PROCESSED_KITTI_360_DIR}
```

Link dataset to codebase.
```bash
# PROCESSED_KITTI_360_DIR: the directory of the processed KITTI-360 dataset (output dir).
ln -s ${PROCESSED_KITTI_360_DIR} ${CODEBASE_DIR}/data/kitti360
```

## 5. ModelNet40

Download [modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and unzip.

Link dataset to the codebase.
```bash
# MODELNET_DIR: the directory of downloaded ModelNet40 dataset.
ln -s ${MODELNET_DIR} ${CODEBASE_DIR}/data/modelnet40_normal_resampled
```