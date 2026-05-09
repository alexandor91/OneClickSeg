# 训练

如果启用 wandb，需要首先执行 `wandb login` 登录

```python
CUDA_VISIBLE_DEVICES=4,5 nohup sh scripts/train.sh -g 2 -d scannet -c intersemseg-spunet-mask3d-15 -n mask3d-25 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u test.py >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=6,7,8 nohup sh scripts/train.sh -g 3 -d scannet -c intersemseg-spunet-mask3d-16 -w /data/shared_workspace/yuzijian/ws/iseg3d/exp/scannet/mask3d-14/model/model_best.pth -n mask3d-16 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1,4,6,7 nohup sh scripts/train.sh -g 4 -d scannet -c intersemseg-spunet-mask3d-15 -w /data/shared_workspace/yuzijian/ws/iseg3d/exp/scannet/mask3d-25/model/model_best.pth -n mask3d-16 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=6,7 nohup sh scripts/train.sh -g 2 -d scannet -c semseg-ptv3 -n semseg-ptv3 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3,4,5 nohup sh scripts/train.sh -g 3 -d scannet -c semseg-ptv3 -n semseg-ptv3 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base-2 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=3,4 nohup sh scripts/train.sh -g 2 -d scannet -c insseg-mask3d-spunet-learn -n insseg-mask3d-spunet-7 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup sh scripts/train.sh -g 4 -d scannet -c insseg-mask3d-spunet-select-fps -n insseg-mask3d-spunet-select-fps-0  > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup sh scripts/train.sh -g 4 -d scannet -c iseg-agile3d-v1m1 -n iseg-agile3d-v1m1-1 > /dev/null 2>&1 &

nohup bash scripts/wait_gpu.sh -g 1,3,4,5 -e cu121_torch251 -d scannet -c iseg-agile3d-v1m1 -n iseg-agile3d-v1m1-2 > /dev/null 2>&1 &
```
<!-- TODO 可学习的背景查询, 用于zero-shot分割背景前景 -->
<!-- 在 MLP head 那里，建立一个 reverse MLP，用于语义查询 -->