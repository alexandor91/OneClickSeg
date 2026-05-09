# 点采样

在完成数据集设置与预处理后(见[dataset.md](./dataset.md))，对点云中的点做采样, 忽略 ignored point

在根目录执行:

```bash
python pointcept/datasets/iseg/sample.py -t fps -d scannet -c iseg-agile3d-v1m1 -n 100
python pointcept/datasets/iseg/sample.py -t fps -d kitti360 -c iseg-agile3d-v1m1 -n 100

参数:
- `-t`: sample 类型
    - fps: farthest point sampling
    - rand: 整体随机采样
    - sem: semantic 内部随机采样
    - ins: instance 内部随机采样
    - super: superpoint 内部随机采样