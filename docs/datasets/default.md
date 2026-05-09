# 默认 dataloader

# datasets.utils

collate_fn(), 合并 minibatch 的数据, 将能 concat 的数据 concat 到一起

point_collate_fn(), 在 collate_fn() 基础上, 执行 [Mix3D](https://arxiv.org/pdf/2110.02210.pdf)
- 按 `mix_prob` 的概率随机混合 mini-batch
- 相邻样本, 两两混合
```python
if random.random() < mix_prob:
    if "offset" in batch.keys():
        batch["offset"] = torch.cat(
            [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
        )
```
注意, 这里只是对mix3d合并的样本做了 instance reid