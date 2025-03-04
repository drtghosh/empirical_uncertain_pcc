import os
import json
import random

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

from data_utils import farthest_point_sampling as fps
from data_utils import positional_encoding


def get_dataloader_buildingpcc(split, config):
    is_shuffle = (split == 'train')

    if config.module == "gen" or config.module == 'imle_gen':
        dataset = BuildingPCCGen(split, config.data_root, config.data_raw_root, config.category, config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = BuildingPCCAE(split, config.data_root, config.category, config.n_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


def split_data(path):
    split_info = {"train": list(), 'validation': list(), 'test': list()}
    with open(path, 'r') as f:
        data_dict = json.loads(f.read())[0]
        split_info["train"] = data_dict["train"]
        split_info["validation"] = data_dict["val"]
        split_info["test"] = data_dict["test"]
    return split_info


def read_point_cloud_ply(path):
    pc = o3d.io.read_point_cloud(path)
    return np.array(pc.points, np.float32)


class BuildingPCCAE(Dataset):
    def __init__(self, split, data_root, data_file, n_pts):
        super(BuildingPCCAE, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.data_file = data_file
        self.data_names, self.data_paths = self._load_data()
        self.n_pts = n_pts

    def __getitem__(self, index):
        ply_path = self.data_paths[index]

        pc = read_point_cloud_ply(ply_path)
        pc = fps(pc, k=self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32)
        enc_pc = positional_encoding(pc).transpose(1, 0)
        pc = pc.transpose(1, 0)
        return {"id": self.data_names[index], "points": pc, "points_encoded": enc_pc}

    def _load_data(self):
        split_dict = split_data(os.path.join(self.data_root, self.data_file))
        split_names = split_dict[self.split]
        split_paths = list()
        for name in split_names:
            ply_path = os.path.join(self.data_root, 'complete', name + '.ply')
            if os.path.exists(ply_path):
                split_paths.append(ply_path)

        return split_names, split_paths

    def __len__(self):
        return len(self.data_paths)


class BuildingPCCGen(Dataset):
    def __init__(self, split, data_root, data_file, n_pts):
        super(BuildingPCCGen, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.data_file = data_file
        self.data_names, self.data_paths = self._load_data()
        self.n_pts = n_pts
        self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        ply_path = self.data_paths[index]

        pc = read_point_cloud_ply(ply_path)
        pc = fps(pc, k=self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32)
        enc_pc = positional_encoding(pc).transpose(1, 0)
        pc = pc.transpose(1, 0)
        return {"id": self.data_names[index], "points": pc, "points_encoded": enc_pc}

    def _load_data(self):
        split_dict = split_data(os.path.join(self.data_root, self.data_file))
        split_names = split_dict[self.split]
        split_paths = list()
        for name in split_names:
            ply_path = os.path.join(self.data_root, 'complete', name + '.ply')
            if os.path.exists(ply_path):
                split_paths.append(ply_path)

        return split_names, split_paths

    def __len__(self):
        return len(self.data_paths)


if __name__ == "__main__":
    pass
