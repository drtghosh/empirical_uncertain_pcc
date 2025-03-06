import os
import json
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_utils import farthest_point_sampling as fps
from data_utils import positional_encoding, add_noise_pc, random_sample
from data_utils import read_point_cloud_las, read_point_cloud_ply


def get_dataloader_pcn(split, config):
    is_shuffle = (split == 'train')

    if config.module == "c_gan" or config.module == 'imle_gen':
        dataset = PCNGen(split, config.data_root, config.data_file, config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = PCNAE(split, config.data_root, config.data_file, config.n_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


def split_data_by_id(path, category):
    split_info = {"train": list(), 'validation': list(), 'test': list()}
    with open(os.path.join(path, 'train.list'), 'r') as f:
        lines_train = f.read().splitlines()
    if category != 'all':
        lines_train = list(filter(lambda x: x.startswith(cat2id[category]), lines_train))
    with open(path, 'r') as f:
        data_dict = json.loads(f.read())[0]
        split_info["train"] = data_dict["train"]
        split_info["validation"] = data_dict["val"]
        split_info["test"] = data_dict["test"]
    return split_info


cat2id = {
    # seen categories
    "airplane": "02691156",  # plane
    "cabinet": "02933112",  # dresser
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "vessel": "04530566",  # boat

    # alis for some seen categories
    "boat": "04530566",  # vessel
    "couch": "04256520",  # sofa
    "dresser": "02933112",  # cabinet
    "plane": "02691156",  # airplane
    "watercraft": "04530566",  # boat

    # unseen categories
    "bus": "02924116",
    "bed": "02818832",
    "bookshelf": "02871439",
    "bench": "02828884",
    "guitar": "03467517",
    "motorbike": "03790512",
    "skateboard": "04225987",
    "pistol": "03948459",
}


class PCNAE(Dataset):
    def __init__(self, split, data_root, data_file, gt_path, n_pts):
        super(PCNAE, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.data_file = data_file
        self.gt_path = os.path.join(data_root, gt_path)
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
            ply_path = os.path.join(self.gt_path, name + '.ply')
            if os.path.exists(ply_path):
                split_paths.append(ply_path)

        return split_names, split_paths

    def __len__(self):
        return len(self.data_names)


class PCNGen(Dataset):
    def __init__(self, split, data_root, data_file, gt_path, partial_path, n_pts):
        super(PCNGen, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.data_file = data_file
        self.gt_path = os.path.join(data_root, gt_path)
        self.partial_path = os.path.join(data_root, partial_path)
        self.data_names, self.gt_paths, self.partial_paths = self._load_data()
        self.n_pts = n_pts
        self.partial_pts = n_pts // 2
        self.rng = random.Random(1234)

    def __getitem__(self, index):
        # read partial cloud
        render_choice = self.rng.randint(0, 1)
        partial_path = self.partial_paths[index].format(render_choice)
        partial_pc = read_point_cloud_las(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32).transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)
        return {"id": self.data_names[index], "gt_points": pc, "partial_id": render_choice,
                "partial_points": partial_pc}

    def _load_data(self):
        split_dict = split_data(os.path.join(self.data_root, self.data_file))
        split_names = split_dict[self.split]
        gt_paths = list()
        partial_paths = list()
        for name in split_names:
            gt_ply_path = os.path.join(self.gt_path, name + '.ply')
            partial_las_path = os.path.join(self.partial_path, name, '0{}.las')
            if os.path.exists(gt_ply_path) and os.path.exists(partial_las_path):
                gt_paths.append(gt_ply_path)
                partial_paths.append(partial_las_path)

        return split_names, gt_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


if __name__ == "__main__":
    pass
