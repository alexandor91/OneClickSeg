import os
import argparse
import torch
import numpy as np
import tqdm
from functools import partial
import sys

sys.path.append(".")
from pointcept.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointcept.datasets import build_dataset
from pointcept.utils_iseg.ins_seg import unique_id
import pointops


def sample(
    dataset_cfg,
    sample_type,
    sample_num,
    sample_func,
    semantic_ignore=-1,
    instance_ignore=-1,
    semantic_background=(0, 1),
):
    dataset = build_dataset(dataset_cfg)
    dataset.loop = 1
    for idx in tqdm.trange(len(dataset)):
        # sample dataset raw data
        data_dict = dataset.get_data(idx)
        save_dir = dataset.data_list[idx]
        sem_ignore_mask = ~np.in1d(
            data_dict["segment"], (semantic_ignore, *semantic_background)
        )
        ins_ignore_mask = data_dict["instance"] != instance_ignore
        ignore_mask = sem_ignore_mask & ins_ignore_mask  # 确保采样的点都是关心点
        if not ignore_mask.any():
            # for test set, all the gt labels are set to be ignore_label
            # set not ignore any point
            ignore_mask[:] = True
        sampled_idx = sample_func(data_dict, sample_num, ignore_mask)
        bg_name = "_bg" if not len(semantic_background) else ""
        file_name = f"sampled_idx_{sample_type}_{sample_num}{bg_name}.npy"
        if sample_type == "center":
            file_name = f"sampled_idx_{sample_type}.npy"
        np.save(os.path.join(save_dir, file_name), sampled_idx)


def fps_sampling(data_dict, sample_num, ignore_mask):
    inverse = torch.where(torch.tensor(ignore_mask).bool().cuda())[0]  # 还原索引
    p = torch.tensor(data_dict["coord"][ignore_mask]).float().cuda().contiguous()
    o = torch.tensor([p.shape[0]]).int().cuda()
    n_o = torch.tensor([sample_num]).int().cuda()
    with torch.no_grad():
        sampled_idx = pointops.farthest_point_sampling(p, o, n_o)
        sampled_idx = inverse[sampled_idx]
    return sampled_idx.cpu().numpy()


def random_sampling(data_dict, sample_num, ignore_mask):
    """全场景随机采样"""
    inverse = torch.where(torch.tensor(ignore_mask).bool())[0]  # 还原索引
    p = torch.tensor(data_dict["coord"][ignore_mask], dtype=torch.float32).contiguous()
    sampled_idx = torch.randperm(p.shape[0], device="cpu")[:sample_num]
    sampled_idx = inverse[sampled_idx]
    return sampled_idx.cpu().numpy()


def fnv_hash_vec_torch(arr):
    """
    FNV64-1A的PyTorch实现版本
    - arr: 形状为 [N, D] 的torch.Tensor，表示需要哈希的D维坐标
    - 返回:shape为 [N] 的torch.Tensor，表示哈希值
    """
    assert arr.dim() == 2
    # 确保使用整数类型
    arr = arr.clone().to(torch.int64)
    # FNV-1a哈希初始值
    FNV_PRIME = torch.tensor(1099511628211, dtype=torch.int64)
    FNV_OFFSET_BASIS = torch.tensor(14695981039346656037, dtype=torch.int64)
    # 初始化哈希数组
    hashed_arr = FNV_OFFSET_BASIS.repeat(arr.shape[0])
    # 对每一维进行哈希
    for j in range(arr.shape[1]):
        hashed_arr = hashed_arr * FNV_PRIME
        # 使用位运算XOR，需要保证操作数在同一设备上
        hashed_arr = torch.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def voxel_sampling(coords, sample_num, mode="center"):
    """voxelize采样, 同一体素内的点取中心点或随机取"""
    coords = torch.tensor(coords, dtype=torch.float32)
    # 计算包围盒
    min_coords = torch.min(coords, dim=0)[0]
    max_coords = torch.max(coords, dim=0)[0]
    bbox = max_coords - min_coords  # 每个维度的范围
    # 估计合适的体素大小，使体素数量接近于样本数量
    voxel_size = float(torch.pow(torch.prod(bbox) / sample_num, 1 / 3))
    # 将坐标缩放到体素空间
    voxel_coords = (coords - min_coords) / voxel_size
    voxel_indices = torch.floor(voxel_coords).long()
    # 使用FNV哈希函数创建唯一的体素ID
    voxel_ids = fnv_hash_vec_torch(voxel_indices)
    unique_voxel_ids = torch.unique(voxel_ids)

    # 为每个体素找到最靠近中心的点
    selected_indices = []
    for i, voxel_id in enumerate(unique_voxel_ids):
        voxel_points = torch.where(voxel_ids == voxel_id)[0]
        if mode == "center":
            points_in_voxel = coords[voxel_points]
            voxel_center = torch.mean(points_in_voxel, dim=0)
            distances = torch.sum((points_in_voxel - voxel_center) ** 2, dim=1)
            closest_point = voxel_points[torch.argmin(distances)]
        elif mode == "random":
            closest_point = voxel_points[torch.randint(0, len(voxel_points), (1,))]
        selected_indices.append(closest_point)

    # 如果体素太多，随机选择一部分
    if len(selected_indices) > sample_num:
        perm = torch.randperm(len(selected_indices))[:sample_num]
        selected_indices = selected_indices[perm]
    return selected_indices


def instance_sampling(data_dict, sample_num, ignore_mask):
    """逐个instance做体素采样，每个体素选择最接近体素中心的点"""
    instance_ids = unique_id(torch.tensor(data_dict["instance"]))
    sampled_idx = []
    sample_num_per_ins = sample_num // len(instance_ids)
    for instance_id in instance_ids:
        # 获取实例点的掩码和坐标
        instance_mask = data_dict["instance"] == instance_id
        instance_mask = instance_mask & ignore_mask
        ins_idx = torch.where(instance_mask)[0]
        if ins_idx.shape[0] > sample_num_per_ins:
            # 获取实例点的坐标
            ins_coords = data_dict["coord"][instance_mask]
            # 应用体素采样
            sampled_local_idx = voxel_sampling(ins_coords, sample_num_per_ins)
            # 转换回原始点云索引
            sampled_global_idx = ins_idx[sampled_local_idx]
            sampled_idx.append(sampled_global_idx)
        else:
            # 如果点数不足，直接使用所有点
            sampled_idx.append(ins_idx)
    sampled_idx = torch.cat(sampled_idx, dim=0)
    return sampled_idx.cpu().numpy()


def ins_center_sampling(data_dict, sample_num, ignore_mask):
    """逐个instance选择instance的中心点"""
    coord = data_dict["coord"]
    instance = data_dict["instance"]
    instance_ids = np.unique(instance[ignore_mask])
    sampled_idx = []
    for instance_id in instance_ids:
        # 获取实例点的掩码和坐标
        instance_mask = instance == instance_id
        instance_mask = instance_mask & ignore_mask
        ins_idx = np.where(instance_mask)[0]
        ins_coords = coord[instance_mask]
        ins_center_coord = ins_coords.mean(0)
        distance = ((ins_center_coord - ins_coords) ** 2).sum(1)
        center_local_idx = distance.argmin()
        # 转换回原始点云索引
        center_global_idx = ins_idx[center_local_idx]
        sampled_idx.append(center_global_idx)
    sampled_idx = np.array(sampled_idx)
    return sampled_idx


SAMPLE_TYPE = {
    "fps": {
        "func": fps_sampling,
        "name": "farthest point",
    },
    "rand": {
        "func": random_sampling,
        "name": "random",
    },
    "center": {
        "func": ins_center_sampling,
        "name": "instance center",
    },
}

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--sample-type", type=str, default="fps", help="采样方式")
    parser.add_argument(
        "-d", "--dataset-name", type=str, default="scannet", help="数据集名称"
    )
    parser.add_argument(
        "-c",
        "--cfg-name",
        type=str,
        default="insseg-mask3d-spunet-0",
        help="配置文件名称, 需要保证data配置正确、semantic_ignore instance_ignore semantic_background",
    )
    parser.add_argument("-n", "--sample-num", type=int, default=200, help="采样点数量")
    parser.add_argument(
        "-b", "--background", action="store_true", help="将墙门等语义背景纳入采样"
    )
    args = parser.parse_args()

    # 使用命令行参数
    sample_type = args.sample_type
    dataset_name = args.dataset_name
    cfg_name = args.cfg_name
    sample_num = args.sample_num
    background = args.background

    assert sample_type in [
        "fps",
        "rand",
        "sem",
        "ins",
        "super",
        "center",  # instance centers
    ], f"unsupported sample type {sample_type}"
    assert dataset_name in [
        "scannet",
        "s3dis",
        "kitti360",
    ], f"unsupported dataset {dataset_name}"

    sample_name = SAMPLE_TYPE[sample_type]["name"]
    sample_func = SAMPLE_TYPE[sample_type]["func"]
    config_file = f"configs/{dataset_name}/{cfg_name}.py"
    cfg = default_config_parser(config_file, None)
    cfg = default_setup(cfg)
    print(
        f"start {sample_name} sampling for {dataset_name} dataset into {sample_num} points"
    )
    print(
        f"settings: \
semantic_ignore={cfg.semantic_ignore}, \
instance_ignore={cfg.instance_ignore}, \
semantic_background={cfg.semantic_background if not background else ()}, \
sample_type={sample_type}, sample_num={sample_num}"
    )
    sample = partial(
        sample,
        semantic_ignore=cfg.semantic_ignore,
        instance_ignore=cfg.instance_ignore,
        semantic_background=cfg.semantic_background if not background else (),
    )
    print("sample train dataset")
    sample(cfg.data.train, sample_type, sample_num, sample_func)
    print("sample val dataset")
    sample(cfg.data.val, sample_type, sample_num, sample_func)
    print("sample test dataset")
    cfg.data.test.split = "test"
    sample(cfg.data.test, sample_type, sample_num, sample_func)
    print("done")
