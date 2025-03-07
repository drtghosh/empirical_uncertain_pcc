import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_utils import farthest_point_sampling as fps
from data_utils import positional_encoding, add_noise_pc, random_sample
from data_utils import read_point_cloud_h5


def get_dataloader_pcn(split, config):
    is_shuffle = (split == 'train')

    if config.module == "c_gan" or config.module == 'imle_gen':
        dataset = C3DGen(split, config.data_root, config.category, 'gt', 'partial', config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = C3DAE(split, config.data_root, config.category, 'gt', config.n_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


def split_data_by_cat(path, category):
    split_info = {"train": list(), "validation": list(), "test": list()}
    # create category id dictionary
    cat2id = dict()
    with open(os.path.join(path, 'synsetoffset2category.txt'), 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        category, cat_id = line.split('\t')
        cat2id[category] = cat_id

    # read all the list files
    with open(os.path.join(path, 'train.list'), 'r') as ftr:
        lines_train = ftr.read().splitlines()
    with open(os.path.join(path, 'validation.list'), 'r') as fv:
        lines_valid = fv.read().splitlines()
    with open(os.path.join(path, 'test.list'), 'r') as fts:
        lines_test = fts.read().splitlines()

    # filter according to the category
    if category != 'all':
        cat_id = cat2id[category]
        lines_train = list(filter(lambda x: x.startswith(cat_id), lines_train))
        lines_valid = list(filter(lambda x: x.startswith(cat_id), lines_valid))
        lines_test = list(filter(lambda x: x.startswith(cat_id), lines_test))

    for line in lines_train:
        cat_id, name = line.split('/')
        split_info["train"].append([cat_id, name])
    for line in lines_valid:
        cat_id, name = line.split('/')
        split_info["validation"].append([cat_id, name])
    for line in lines_test:
        cat_id, name = line.split('/')
        split_info["test"].append([cat_id, name])

    return split_info


class C3DAE(Dataset):
    def __init__(self, split, data_root, category, gt_path, n_pts):
        super(C3DAE, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.data_names, self.data_paths = self._load_data(category)
        self.n_pts = n_pts

    def __getitem__(self, index):
        h5_path = self.data_paths[index]

        pc = read_point_cloud_h5(h5_path)
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
            h5_path = os.path.join(self.gt_path, name[0], name[1] + '.h5')
            if os.path.exists(h5_path):
                split_paths.append(h5_path)

        return split_names, split_paths

    def __len__(self):
        return len(self.data_names)


class C3DGen(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, n_pts):
        super(C3DGen, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.data_names, self.gt_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # read partial cloud
        partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_h5(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32).transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_h5(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)
        return {"id": "/".join(self.data_names[index]), "gt_points": pc, "partial_id": 0,
                "partial_points": partial_pc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        gt_paths = list()
        partial_paths = list()
        for name in split_names:
            gt_h5_path = os.path.join(self.gt_path, name[0], name[1] + '.h5')
            partial_h5_path = os.path.join(self.partial_path, name[0], name[1] + '.h5')
            if os.path.exists(gt_h5_path) and os.path.exists(partial_h5_path):
                gt_paths.append(gt_h5_path)
                partial_paths.append(partial_h5_path)

        return split_names, gt_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


if __name__ == "__main__":
    pass
