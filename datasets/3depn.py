import os
import json
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_utils import farthest_point_sampling as fps
from data_utils import positional_encoding, add_noise_pc, random_sample
from data_utils import read_point_cloud_ply


def get_dataloader_3depn(split, config):
    is_shuffle = (split == 'train')

    if config.module == "c_gan" or config.module == 'imle_gen':
        dataset = EPNGen(split, config.data_root, config.category, 'complete', 'partial', config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = EPNAE(split, config.data_root, config.category, 'complete', config.n_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


def get_cat2id(path):
    cat2id = dict()
    with open(path, 'r') as f:
        data_dict = json.loads(f.read())
        for dct in data_dict:
            cat2id[dct['taxonomy_name']] = dct['taxonomy_id']
    return cat2id


def split_data_by_cat(path, category):
    split_info = {"train": list(), "test": list()}

    # read the data file
    with open(path, 'r') as f:
        data_dict = json.loads(f.read())

    # filter according to the category
    if category == 'all':
        for d in data_dict:
            train_names = d['train']['complete']
            category = d['taxonomy_name']
            for n, name in enumerate(train_names):
                train_names[n] = [category, name]
            test_names = d['test']['complete']
            for n, name in enumerate(test_names):
                test_names[n] = [category, name]
            split_info["train"] += train_names
            split_info["test"] += test_names
    else:
        split_info["train"] = list(filter(lambda d: d['taxonomy_name'] == category, data_dict))[0]['train']['complete']
        split_info["test"] = list(filter(lambda d: d['taxonomy_name'] == category, data_dict))[0]['test']['complete']

    return split_info


class EPNAE(Dataset):
    def __init__(self, split, data_root, data_file, category, gt_path, n_pts):
        super(EPNAE, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.data_file = data_file
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.data_names, self.data_paths = self._load_data(category)
        self.n_pts = n_pts

    def __getitem__(self, index):
        ply_path = self.data_paths[index]

        pc = read_point_cloud_ply(ply_path)
        pc = fps(pc, k=self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32)
        enc_pc = positional_encoding(pc).transpose(1, 0)
        pc = pc.transpose(1, 0)
        return {"id": "/".join(self.data_names[index]), "points": pc, "points_encoded": enc_pc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        split_paths = list()
        for name in split_names:
            ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
            if os.path.exists(ply_path):
                split_paths.append(ply_path)

        return split_names, split_paths

    def __len__(self):
        return len(self.data_names)


class EPNGen(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, n_pts):
        super(EPNGen, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.data_names, self.gt_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        self.partial_pts = n_pts // 2
        self.rng = random.Random(1234)

    def __getitem__(self, index):
        # read partial cloud
        render_choice = self.rng.randint(0, 7)
        partial_path = self.partial_paths[index].format(render_choice)
        partial_pc = read_point_cloud_ply(partial_path)
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
        return {"id": "/".join(self.data_names[index]), "gt_points": pc, "partial_id": render_choice,
                "partial_points": partial_pc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        gt_paths = list()
        partial_paths = list()
        for name in split_names:
            gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
            if self.split == 'train':
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '_{}.ply')
            else:
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '.ply')
            if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
                gt_paths.append(gt_ply_path)
                partial_paths.append(partial_ply_path)

        return split_names, gt_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


if __name__ == "__main__":
    pass
