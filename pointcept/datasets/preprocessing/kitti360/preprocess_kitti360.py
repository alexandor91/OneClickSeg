"""
Preprocessing Script for KITTI-360
Converts KITTI-360 labels to windows in numpy format.

Author: Zijian Yu (https://github.com/yzj2019)
Modified from https://github.com/autonomousvision/kitti360Scripts
"""

import os
from tqdm import tqdm
from pathlib import Path
import glob, numpy as np, multiprocessing as mp, torch, json, argparse

os.chdir(Path(os.path.dirname(__file__)))
from helpers.labels import id2label
from helpers.ply import read_ply


def f_test(data_path, output_dir):
    data = read_ply(data_path)
    points = np.vstack((data["x"], data["y"], data["z"])).T
    colors = np.vstack((data["red"], data["green"], data["blue"])).T
    colors = colors.astype(np.float32)
    np.save(os.path.join(output_dir, "coord.npy"), points)
    np.save(os.path.join(output_dir, "color.npy"), colors)


def f(data_path, output_dir):
    data = read_ply(data_path)
    points = np.vstack((data["x"], data["y"], data["z"])).T
    colors = np.vstack((data["red"], data["green"], data["blue"])).T
    colors = colors.astype(np.float32)

    """semantic"""
    # 记得修改 helpers/labels.py 中的 trainid
    # (譬如 kitti360Scripts 中忽略 255, 这里改为 -1, 将原本-1所在的 license plate 改为 19)
    ignore_label = -1
    sem_labels_raw = data["semantic"]
    sem_labels = np.ones_like(sem_labels_raw) * ignore_label
    for i in np.unique(sem_labels_raw):
        sem_labels[sem_labels_raw == i] = id2label[i].trainId

    """instance"""
    instance_labels_raw = data["instance"]
    instance_labels = np.ones_like(instance_labels_raw) * ignore_label
    # unique instance id (regardless of semantic label)
    ins_cnt = 0
    ins_map = {}
    for i, ins_id in enumerate(np.unique(instance_labels_raw)):
        if ins_id % 1000 == 0:
            instance_labels[instance_labels_raw == ins_id] = ignore_label
        else:
            instance_labels[instance_labels_raw == ins_id] = ins_cnt
            ins_map[ins_id] = ins_cnt
            ins_cnt += 1
    instance_labels[sem_labels == ignore_label] = ignore_label

    np.save(os.path.join(output_dir, "coord.npy"), points)
    np.save(os.path.join(output_dir, "color.npy"), colors)
    np.save(os.path.join(output_dir, "segment.npy"), sem_labels)
    np.save(os.path.join(output_dir, "instance.npy"), instance_labels)


if __name__ == "__main__":
    # TODO 多线程
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="path to KITTI-360 dataset"
    )
    parser.add_argument(
        "--output_root", required=True, help="path to save processed data"
    )
    opt = parser.parse_args()

    for split in ["train", "val"]:
        kitti360Path = opt.dataset_root
        with open(
            os.path.join(
                kitti360Path,
                "data_3d_semantics",
                "train",
                f"2013_05_28_drive_{split}.txt",
            ),
            "r",
        ) as file_handle:
            files = file_handle.read().split()
        print(f"Processing {split} split")
        for fn in tqdm(files):
            data_path = os.path.join(kitti360Path, fn)
            drive_id = os.path.dirname(fn).split("/")[-2]
            file_name = os.path.basename(fn).split(".")[0]
            output_dir = os.path.join(opt.output_root, split, drive_id, file_name)
            os.makedirs(output_dir, exist_ok=True)
            f(data_path, output_dir)

    split = "test"
    print(f"Processing {split} split")
    files = sorted(
        glob.glob(
            os.path.join(
                kitti360Path, "data_3d_semantics", split, "*", "static", "*.ply"
            )
        )
    )
    for fn in tqdm(files):
        data_path = fn
        drive_id = os.path.basename(os.path.split(os.path.dirname(fn))[0])
        file_name = os.path.basename(fn).split(".")[0]
        output_dir = os.path.join(opt.output_root, split, drive_id, file_name)
        os.makedirs(output_dir, exist_ok=True)
        f_test(data_path, output_dir)
