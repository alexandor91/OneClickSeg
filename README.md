# ISeg3D — Interactive 3D Segmentation Framework

**ISeg3D** is a unified codebase for **interactive, instance, and semantic segmentation of 3D point clouds**, built on top of [Pointcept](https://github.com/Pointcept/Pointcept). It implements click-driven interactive segmentation with strong modern backbones (SparseUNet, Point Transformer v1/v2/v3, OctFormer, Swin3D, Sonata, …) and ships with ready-to-use configurations for ScanNet, S3DIS, KITTI-360, NuScenes, SemanticKITTI, Matterport3D, Waymo, ModelNet40, ScanNet++, and Structured3D.

> AI-generated docs are also available at <https://deepwiki.com/yzj2019/ISeg3D>.

---

## Table of contents

- [Highlights](#highlights)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)
- [Adding new models / datasets](#adding-new-models--datasets)
- [Docker](#docker)
- [Citation & acknowledgments](#citation--acknowledgments)
- [License](#license)

---

## Highlights

- **Interactive segmentation** — click-based segmentation with simulated and real clicks. Includes:
  - `Clicker` modules (2D and 3D variants) under `pointcept/utils_iseg/clicker.py`
  - Click-sampling strategies (FPS / random / semantic / instance / superpoint) in `pointcept/datasets/iseg/sample.py`
  - `HungarianMatcher` for query↔mask assignment (`pointcept/utils_iseg/matcher.py`)
  - `Scene` / `Query` structures for multi-resolution feature fusion (`pointcept/utils_iseg/structure.py`)
- **Multi-task, multi-backbone**: interactive (Agile3D, ISeg3D, InterObj), instance (Mask3D, PointGroup), semantic (PTv1/v2/v3, SparseUNet, MinkUNet, SPVCNN, Sonata, Swin3D, OctFormer, Stratified Transformer, OACNN), classification (ModelNet40).
- **One config = one experiment** — MMCV-style python configs in `configs/<dataset>/` inheriting from `configs/_base_/`.
- **Distributed-ready** — multi-GPU and multi-machine launcher with auto SLURM detection (`pointcept/engines/launch.py`).
- **Mix3D, AMP, Wandb, hooks** — Mix3D augmentation, automatic mixed precision, Weights & Biases logging, and a pluggable hook system (checkpointing, evaluation, timers, precise evaluator).
- **Reproducible runs** — every experiment automatically snapshots `scripts/`, `tools/`, and `pointcept/` into `exp/<dataset>/<exp_name>/code/`.

## Repository layout

```
ISeg3D/
├── tools/                        # entry points
│   ├── train.py                  # training
│   ├── test.py                   # evaluation
│   ├── test_s3dis_6fold.py       # S3DIS 6-fold evaluation
│   └── send_mail.py              # job-completion email helper
├── scripts/                      # shell wrappers
│   ├── train.sh / test.sh        # multi-GPU launchers
│   ├── wait_gpu.sh               # queue training behind running GPU jobs
│   └── build_image.sh            # docker image build
├── configs/                      # MMCV-style experiment configs
│   ├── _base_/                   # default runtimes & dataset stubs
│   ├── scannet/, s3dis/, kitti360/, nuscenes/, ...
│   └── sonata/                   # Sonata pretraining + fine-tuning recipes
├── pointcept/                    # core library
│   ├── datasets/                 # dataset loaders
│   │   ├── iseg/                 # interactive-seg dataset, sampling, transforms
│   │   ├── preprocessing/        # raw → processed scripts (ScanNet, KITTI-360, etc.)
│   │   └── transform.py          # GridSample, augmentations
│   ├── models/                   # backbones + heads
│   │   ├── agile3d/              # interactive Mask3D-style decoder
│   │   ├── iseg3d/               # interactive segmentor + mask decoder
│   │   ├── interobj/             # interactive object segmentation
│   │   ├── mask3d/, point_group/ # instance segmentation
│   │   ├── point_transformer*/   # PTv1, v2, v3
│   │   ├── sparse_unet/, spvcnn/ # voxel backbones
│   │   ├── sonata/               # self-supervised pretraining
│   │   └── losses/               # CE, BCE, dice, focal, lovasz, …
│   ├── engines/                  # trainers / testers
│   │   ├── train.py, test.py     # base classes + registry
│   │   ├── iseg.py               # InsSegTrainer (interactive / instance)
│   │   ├── defaults.py           # arg parser, config parser, default setup
│   │   ├── launch.py             # distributed launch
│   │   └── hooks/                # checkpoint, evaluator, timer, wandb
│   ├── utils_iseg/               # interactive-segmentation primitives
│   │   ├── clicker.py            # 2D/3D click simulation
│   │   ├── sample_clicks.py      # click-sampling strategies
│   │   ├── matcher.py            # Hungarian matching
│   │   ├── structure.py          # Scene & Query containers
│   │   └── ins_seg.py, plot.py   # mask↔ID, post-processing, visualization
│   └── utils/                    # registry, comm, logger, misc
├── libs/                         # CUDA / C++ extensions (build from source)
│   ├── pointops/                 # KNN, ball query, grouping, sampling
│   ├── pointops2/                # alternative pointops impl
│   ├── pointgroup_ops/           # PointGroup clustering ops
│   └── scannet_segmentator/      # superpoint segmentation (git submodule)
├── docker/                       # Dockerfile + build resources
├── docs/                         # markdown documentation
├── example/                      # notebooks (dataloader.ipynb, vis.ipynb, ckpt.ipynb)
├── environment.yml               # conda environment spec
└── LICENSE                       # MIT (Pointcept upstream)
```

## Installation

> Tested on Ubuntu 20.04 / 22.04, CUDA 11.7–12.4, PyTorch 2.0+, Python 3.10–3.11. The recommended combo is **CUDA 12.1 + PyTorch 2.5.1** (RTX 30/40 series).

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/yzj2019/ISeg3D.git
cd ISeg3D
```

### 2. Create the conda environment

The fastest path uses the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
```

Or install manually (matches `docs/envs.md` more closely; pick one that matches your hardware):
```bash
conda create -n iseg3d python=3.11 -y
conda activate iseg3d

# PyTorch (pick the CUDA version that matches your driver)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Pointcept common deps
conda install ninja h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx wandb yapf addict einops \
              scipy plyfile termcolor timm matplotlib ipykernel \
              -c conda-forge -y
pip install open3d torch-geometric easydict opencv-python

# PyG extensions (must match torch + cuda)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Sparse convolutions
pip install spconv-cu121

# Flash attention
MAX_JOBS=4 pip install flash-attn==2.3.0 --no-build-isolation
```

### 3. Build the CUDA extensions

Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (e.g. `7.5` = RTX 20xx, `8.0` = A100, `8.6` = RTX 30xx, `8.9` = RTX 40xx — see [NVIDIA's compute-capability table](https://developer.nvidia.com/cuda-gpus)):

```bash
# pointops
cd libs/pointops
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python setup.py install
cd ../..

# pointgroup_ops
conda install -c bioconda google-sparsehash -y
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..

# Superpoint segmentator (git submodule)
cd libs/scannet_segmentator
pip install .
cd ../..
```

For the full installation walk-through (and troubleshooting `flash-attn` / `spconv` / `pointops` errors), see **[`docs/envs.md`](docs/envs.md)**.

## Data preparation

Each dataset has its own preprocessing pipeline. Concrete commands for **ScanNet, S3DIS, SemanticKITTI, KITTI-360, ModelNet40**, and others are in **[`docs/datasets/preprocessing.md`](docs/datasets/preprocessing.md)**. The general pattern:

1. Download the raw dataset.
2. Run the corresponding `pointcept/datasets/preprocessing/<dataset>/preprocess_*.py`.
3. Symlink the processed directory into `data/<dataset>` at the repo root:
   ```bash
   ln -s ${PROCESSED_DIR} ${REPO_ROOT}/data/<dataset>
   ```

Example (ScanNet v2):
```bash
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py \
       --dataset_root ${RAW_SCANNET_DIR} \
       --output_root  ${PROCESSED_SCANNET_DIR}
ln -s ${PROCESSED_SCANNET_DIR} ${REPO_ROOT}/data/scannet
```

### Click sampling for interactive segmentation

After preprocessing, simulate the clicks used at training/eval time:

```bash
python pointcept/datasets/iseg/sample.py -t fps  -d scannet  -c iseg-agile3d-v1m1 -n 100
python pointcept/datasets/iseg/sample.py -t fps  -d kitti360 -c iseg-agile3d-v1m1 -n 100
```

Supported click-sampling strategies (`-t`): `fps` (farthest-point), `rand` (uniform random), `sem` (semantic-aware), `ins` (instance-aware), `super` (superpoint-aware). See **[`docs/datasets/sample.md`](docs/datasets/sample.md)**.

## Training

The standard launcher is `scripts/train.sh`:

```bash
bash scripts/train.sh -g <NUM_GPU> -d <DATASET> -c <CONFIG> -n <EXP_NAME>
```

Common flags:

| Flag | Meaning |
| ---- | ------- |
| `-g` | GPUs to use (default: all visible) |
| `-d` | dataset name (must match `configs/<dataset>/`) |
| `-c` | config file basename under `configs/<dataset>/` (no `.py`) |
| `-n` | experiment name (used as `exp/<dataset>/<EXP_NAME>/`) |
| `-w` | path to a pretrained weight (`.pth`) to load |
| `-r` | `true` to resume from `model_last.pth` |
| `-m` | number of machines (multi-node) |

### Examples

```bash
# Interactive segmentation (Agile3D on ScanNet, 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/train.sh -g 4 -d scannet -c iseg-agile3d-v1m1 -n iseg-agile3d-v1m1-1

# Semantic segmentation with PTv3 on ScanNet, 2 GPUs, in the background
CUDA_VISIBLE_DEVICES=4,5 \
  nohup bash scripts/train.sh -g 2 -d scannet -c semseg-pt-v3m1-0-base -n ptv3-baseline \
  > /dev/null 2>&1 &

# Resume a run from its last checkpoint
bash scripts/train.sh -g 4 -d scannet -c iseg-agile3d-v1m1 -n iseg-agile3d-v1m1-1 -r true

# Fine-tune from a pretrained weight
bash scripts/train.sh -g 4 -d scannet -c iseg-agile3d-v1m1 -n iseg-finetune \
                      -w exp/scannet/<previous_run>/model/model_best.pth

# Queue behind a running GPU job (waits for the GPU to free up, then trains)
nohup bash scripts/wait_gpu.sh -g 1,3,4,5 -e iseg3d \
                               -d scannet -c iseg-agile3d-v1m1 -n iseg-queued \
      > /dev/null 2>&1 &
```

`wait_gpu.sh` additionally sends a notification email when the GPU is acquired — set `QQ_EMAIL_USER` and `QQ_EMAIL_PASS` first.

To enable [Weights & Biases](https://wandb.ai/) logging, run `wandb login` once before training. See **[`docs/train.md`](docs/train.md)** for additional examples.

## Testing

After training, run evaluation against the best (or any) checkpoint:

```bash
bash scripts/test.sh -d <DATASET> -c <CONFIG> -n <EXP_NAME> -w model_best
```

`<EXP_NAME>` selects the experiment folder under `exp/<DATASET>/`; `<WEIGHT>` is the basename of the checkpoint inside `exp/<DATASET>/<EXP_NAME>/model/` (e.g. `model_best`, `model_last`).

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/test.sh -d scannet -c iseg-agile3d-v1m1 -n iseg-agile3d-v1m1-1 \
                       -w model_best
```

For S3DIS 6-fold cross-validation use `tools/test_s3dis_6fold.py`. See **[`docs/test.md`](docs/test.md)**.

## Visualization

Tensorboard:
```bash
tensorboard --logdir=exp/scannet/<EXP_NAME> --port=8001
```
> If you SSH into a remote machine, disable any reverse SSH tunnel for that port to avoid forwarding loops.

Notebooks: `example/vis.ipynb` (point-cloud rendering), `example/dataloader.ipynb` (dataset/transform inspection), and `example/ckpt.ipynb` (checkpoint exploration). See **[`docs/vis.md`](docs/vis.md)**.

## Adding new models / datasets

ISeg3D uses Pointcept's `Registry` pattern. Quick recipe:

1. **Implement** the new model under `pointcept/models/<your_model>/` and decorate the class with `@MODELS.register_module("YourModelName")`.
2. **Import** the package in `pointcept/models/__init__.py` so the registry sees it.
3. **Write a config** in `configs/<dataset>/` referencing the new model via its registered name.

Datasets work the same way (`@DATASETS.register_module(...)`), with loaders in `pointcept/datasets/`. Custom transforms go in `pointcept/datasets/transform.py` (or `pointcept/datasets/iseg/transform.py` for interactive-only ones).

For details on the registry and config conventions, read **[`docs/registry.md`](docs/registry.md)** and **[`docs/config.md`](docs/config.md)**.

## Docker

A reference image is provided in `docker/Dockerfile` (CUDA 11.6 / cuDNN 8 / Ubuntu 20.04 base). Build it with:

```bash
bash scripts/build_image.sh -t 2.0.1 -c 11.7 --cudnn 8
```

`-t` / `-c` / `--cudnn` select the PyTorch / CUDA / cuDNN versions inside the image.

## Citation & acknowledgments

This repository is built on **[Pointcept](https://github.com/Pointcept/Pointcept)** and incorporates ideas from [Mask3D](https://arxiv.org/abs/2210.03105), [Agile3D](https://arxiv.org/abs/2306.00977), [EMC-Click](https://github.com/feiaxyt/EMC-Click), and [Sonata](https://github.com/Pointcept/Sonata). If you use ISeg3D, please cite the original Pointcept work and the model whose recipe you followed.

## License

Released under the **MIT License** (see [`LICENSE`](LICENSE)). Submodules and third-party code retain their original licenses.

---

*Maintainer: [Zijian Yu](https://github.com/yzj2019). Issues and pull requests welcome.*