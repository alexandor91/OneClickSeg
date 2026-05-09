#!/bin/sh
mkdir -p ~/miniconda3
# curl -fSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# 网络不好时，使用本地miniconda.sh
bash /tmp/resources/miniconda.sh -b -u -p ~/miniconda3

# 初始化conda
~/miniconda3/bin/conda init bash

# 换源
# 1. 生成默认配置文件（如果不存在）
conda config --set show_channel_urls yes
# 2. 添加清华源（顺序影响优先级）
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# 3. 设置通道优先级策略
conda config --set channel_priority flexible

# 换base环境里的pip源
pip config set global.index-url http://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/