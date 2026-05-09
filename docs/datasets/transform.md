
-[ ] 现在的 InstanceParser 和 DefaultDataset.prepare_test_data() 有冲突，所以暂时只用 train mode

# GridSample 类详细分析

GridSample 是一个用于点云数据处理的变换类，主要功能是将点云数据按照网格进行采样。该类在 `pointcept/datasets/transform.py` 中定义。

## 初始化参数

```python
def __init__(
    self,
    grid_size=0.05,
    hash_type="fnv",
    mode="train",
    keys=("coord", "color", "normal", "segment"),
    return_inverse=False,
    return_grid_coord=False,
    return_min_coord=False,
    return_displacement=False,
    project_displacement=False,
):
```

- `grid_size`: 网格大小，默认为0.05，用于将点云坐标缩放到网格空间
- `hash_type`: 哈希函数类型，可选"fnv"或其他（使用ravel_hash_vec）
- `mode`: 模式，可选"train"或"test"
- `keys`: 需要处理的数据字典中的键
- `return_inverse`: 是否返回逆映射
- `return_grid_coord`: 是否返回网格坐标
- `return_min_coord`: 是否返回最小坐标
- `return_displacement`: 是否返回位移
- `project_displacement`: 是否投影位移

## 核心处理逻辑

1. **坐标缩放和网格化**:
   ```python
   scaled_coord = data_dict["coord"] / np.array(self.grid_size)
   grid_coord = np.floor(scaled_coord).astype(int)
   ```
   将原始坐标除以网格大小，然后向下取整得到网格坐标。

2. **坐标归一化**:
   ```python
   min_coord = grid_coord.min(0)
   grid_coord -= min_coord
   scaled_coord -= min_coord
   min_coord = min_coord * np.array(self.grid_size)
   ```
   减去最小坐标使所有坐标非负，便于哈希处理。

3. **哈希和排序**:
   ```python
   key = self.hash(grid_coord)
   idx_sort = np.argsort(key)
   key_sort = key[idx_sort]
   _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
   ```
   对网格坐标进行哈希，然后排序并找出唯一值。

4. **训练模式下的采样**:
   在训练模式下，每个网格中随机选择一个点：
   ```python
   idx_select = (
       np.cumsum(np.insert(count, 0, 0)[0:-1])
       + np.random.randint(0, count.max(), count.size) % count
   )
   idx_unique = idx_sort[idx_select]
   ```

5. **测试模式下的处理**:
   在测试模式下，为每个网格中的点创建单独的数据部分。
   返回的 index 是原数据向 voxel 的映射 index

## 哈希函数

类提供了两种哈希方法：
- `ravel_hash_vec`: 基于坐标的线性索引哈希
- `fnv_hash_vec`: FNV64-1A哈希算法，通常更均匀分布

## 从新数据映射回原数据

如果需要在做了GridSample之后从新数据映射到原数据，可以使用inverse索引

直接使用 `reconstructed = before_reconstruct[inverse]` 即可，inverse的产生过程示例如下：
```text
原始数组: [1 1 2 3 3 3 5 5 8]
唯一值: [1 2 3 5 8]
逆映射索引: [0 0 1 2 2 2 3 3 4]
```