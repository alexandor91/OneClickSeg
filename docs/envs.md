# 环境配置

## 1. Example

for RTX 3090, use cuda=12.1, pytorch=2.1.0, spconv-cu120, TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6".
```bash
conda create -n cu121_torch251 python=3.11 -y
conda activate cu121_torch251
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# pytorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# pointcept common
conda install ninja -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm ipykernel matplotlib -c conda-forge -y

# pyg, specify torch and cuda version
pip install open3d torch-geometric easydict opencv-python
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# spconv, requires python <= 3.11, see https://github.com/traveller59/spconv#prebuilt
pip install spconv-cu121

# Flash attention
MAX_JOBS=4 pip install flash-attn==2.3.0 --no-build-isolation

# pointops
cd libs/pointops
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python  setup.py install
cd ../..
# pointgroup_ops
conda install -c bioconda google-sparsehash -y
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..
# superpoint
cd libs/scannet_segmentator
pip install .
cd ../..

# for pytorch <= 2.1.*, downgrade numpy
# conda install numpy=1.26.4 -y
```

## 2. Each Part

### 2.1. PyTorch

PTv3 need cuda >= 11.7, pytorch >= 1.12.0

Choose version here: https://pytorch.org/get-started/previous-versions/ , with `nvcc -V`, e.g.

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 2.2. PyG

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 2.2. SpConv

for SparseUNet, refer https://github.com/traveller59/spconv, version must be satisfied with cudatoolkit version

### 2.3. Flash Attention

see https://github.com/Dao-AILab/flash-attention

`undefined symbol` 错误，通常是版本不匹配

寻找对应的版本：去到 github 库 的 [publish.yml](https://github.com/Dao-AILab/flash-attention/blob/v2.3.0/.github/workflows/publish.yml) 文件，确认支持的 pytorch 和 cuda 版本
```yml
torch-version: ['1.12.1', '1.13.1', '2.0.1', '2.1.0.dev20230731']
cuda-version: ['11.6.2', '11.7.1', '11.8.0', '12.1.0', '12.2.0']
```

查看排除项，确定哪些组合不受支持：
```yml
exclude:
    # Pytorch <= 1.12 does not support Python 3.11
    - torch-version: '1.12.1'
      python-version: '3.11'
    ...
```

### 2.4. libs/pointops

usually, `python setup.py install`

for docker & multi GPU arch:
```bash
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python  setup.py install
```
- e.g. 7.5: RTX 3000; 8.0: A100; 8.6: 3090
- You should modify this to match your GPU compute capability, see https://developer.nvidia.cn/zh-cn/cuda-gpus#compute

## 3. 代码格式化
```bash
conda create -n black python=3.12 -y
conda activate black
pip install black[colorama]==25.1.0
```