import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .data_utils import farthest_point_sampling as fps
from .data_utils import positional_encoding, add_noise_pc, random_sample
from .data_utils import read_point_cloud_ply


def get_dataloader_pcn(split, config):
    is_shuffle = (split == 'train')

    if config.module == "c_gan" or config.module == 'imle_gen' or config.module == 'ebm_gen' or config.module == 'dropout_gen' or config.module == 'drop_con_gen' or config.module == 'ensemble_gen':
        dataset = PCNGen(split, config.data_root, config.category, 'complete', 'partial', config.n_pts,
                         config.partial_pts)
    elif config.module == "ae" or config.module == "vae" or config.module == "vqvae":
        dataset = PCNAE(split, config.data_root, config.category, 'complete', config.n_pts)
    elif config.module == "contrast_ae" or config.module == "contrast_vae" or config.module == "contrast_vqvae":
        dataset = PCNCon(split, config.data_root, config.category, 'complete', 'partial', 'negative', config.n_pts,
                         config.partial_pts)
    elif config.module == "contrast_all_ae" or config.module == "contrast_all_vae" or config.module == "contrast_all_vqvae":
        dataset = PCNConAll(split, config.data_root, config.category, 'complete', 'partial', 'negative', config.n_pts,
                            config.partial_pts)
    elif config.module == "contrast_grid_ae" or config.module == "contrast_grid_vae" or config.module == "contrast_grid_vqvae":
        dataset = PCNConGrid(split, config.data_root, config.category, 'complete', 'partial', 'negative_grid',
                            config.n_pts, config.partial_pts)
    elif config.module == "hessian":
        dataset = PCNHess(split, config.data_root, config.category, 'complete', 'partial', config.n_pts,
                         config.partial_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


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


def split_data_by_cat(path, category):
    split_info = {"train": list(), "validation": list(), "test": list(), "test_novel": list()}

    # read all the list files
    with open(os.path.join(path, 'train.list'), 'r') as ftr:
        lines_train = ftr.read().splitlines()
    with open(os.path.join(path, 'validation.list'), 'r') as fv:
        lines_valid = fv.read().splitlines()
    with open(os.path.join(path, 'test.list'), 'r') as fts:
        lines_test = fts.read().splitlines()
    with open(os.path.join(path, 'test_novel.list'), 'r') as fn:
        lines_novel = fn.read().splitlines()

    # filter according to the category
    if category != 'all':
        cat_id = cat2id[category]
        lines_train = list(filter(lambda x: x.startswith(cat_id), lines_train))
        lines_valid = list(filter(lambda x: x.startswith(cat_id), lines_valid))
        lines_test = list(filter(lambda x: x.startswith(cat_id), lines_test))
        lines_novel = list(filter(lambda x: x.startswith(cat_id), lines_novel))

    for line in lines_train:
        cat_id, name = line.split('/')
        split_info["train"].append([cat_id, name])
    for line in lines_valid:
        cat_id, name = line.split('/')
        split_info["validation"].append([cat_id, name])
    for line in lines_test:
        cat_id, name = line.split('/')
        split_info["test"].append([cat_id, name])
    for line in lines_novel:
        cat_id, name = line.split('/')
        split_info["test_novel"].append([cat_id, name])

    return split_info


class PCNAE(Dataset):
    def __init__(self, split, data_root, category, gt_path, n_pts):
        super(PCNAE, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
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


class PCNGen(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, n_pts, partial_pts):
        super(PCNGen, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.rng = random.Random(1234)
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.data_names, self.gt_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        if partial_pts is not None:
            self.partial_pts = partial_pts
        else:
            self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # read partial cloud
        render_choice = 0
        if self.split == 'train':
            render_choice = self.rng.randint(0, 7)
            partial_path = self.partial_paths[index].format(render_choice)
        else:
            partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_ply(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32)
        partial_enc = positional_encoding(partial_pc).transpose(1, 0)
        partial_pc = partial_pc.transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32)
        pc_enc = positional_encoding(pc).transpose(1, 0)
        pc = pc.transpose(1, 0)
        return {"id": "/".join(self.data_names[index]), "gt_points": pc, "gt_encoded": pc_enc,
                "partial_id": render_choice, "partial_points": partial_pc, "partial_encoded": partial_enc}

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
            # if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
            gt_paths.append(gt_ply_path)
            partial_paths.append(partial_ply_path)

        return split_names, gt_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


class PCNCon(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, negative_path, n_pts, partial_pts):
        super(PCNCon, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.rng = random.Random(1234)
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.negative_path = os.path.join(data_root, split, negative_path)
        self.data_names, self.gt_paths, self.negative_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        if partial_pts is not None:
            self.partial_pts = partial_pts
        else:
            self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # read partial cloud
        render_choice = 0
        if self.split == 'train':
            render_choice = self.rng.randint(0, 7)
            partial_path = self.partial_paths[index].format(render_choice)
        else:
            partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_ply(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32)
        partial_enc = positional_encoding(partial_pc).transpose(1, 0)
        partial_pc = partial_pc.transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)
        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)

        if self.split == 'test':
            return {"id": "/".join(self.data_names[index]), "gt_points": pc,
                    "partial_id": render_choice, "partial_points": partial_pc, "partial_encoded": partial_enc}
        else:
            # read negative (not on surface) cloud
            negative_path = self.negative_paths[index]
            npc = read_point_cloud_ply(negative_path)
            # sample from negative cloud
            npc = random_sample(npc, self.n_pts)
            npc = torch.tensor(npc, dtype=torch.float32).transpose(1, 0)

            return {"id": "/".join(self.data_names[index]), "gt_points": pc, "negative_points": npc,
                    "partial_id": render_choice, "partial_points": partial_pc, "partial_encoded": partial_enc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        gt_paths = list()
        negative_paths = list()
        partial_paths = list()
        for name in split_names:
            gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
            negative_ply_path = os.path.join(self.negative_path, name[0], name[1] + '.ply')
            if self.split == 'train':
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '_{}.ply')
            else:
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '.ply')
            # if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
            gt_paths.append(gt_ply_path)
            negative_paths.append(negative_ply_path)
            partial_paths.append(partial_ply_path)

        return split_names, gt_paths, negative_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


class PCNConAll(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, negative_path, n_pts, partial_pts):
        super(PCNConAll, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.negative_path = os.path.join(data_root, split, negative_path)
        self.data_names, self.gt_paths, self.negative_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        if partial_pts is not None:
            self.partial_pts = partial_pts
        else:
            self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # read partial cloud
        partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_ply(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32)
        partial_enc = positional_encoding(partial_pc).transpose(1, 0)
        partial_pc = partial_pc.transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)
        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)

        if self.split == 'test':
            return {"id": "/".join(self.data_names[index]), "gt_points": pc,
                    "partial_id": 0, "partial_points": partial_pc, "partial_encoded": partial_enc}
        else:
            # read negative (not on surface) cloud
            negative_path = self.negative_paths[index]
            npc = read_point_cloud_ply(negative_path)
            # sample from negative cloud
            npc = random_sample(npc, self.n_pts)
            npc = torch.tensor(npc, dtype=torch.float32).transpose(1, 0)

            return {"id": "/".join(self.data_names[index]), "gt_points": pc, "negative_points": npc,
                    "partial_id": index % 8, "partial_points": partial_pc, "partial_encoded": partial_enc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        gt_paths = list()
        negative_paths = list()
        partial_paths = list()
        for name in split_names:
            if self.split == 'train':
                for i in range(8):
                    gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
                    negative_ply_path = os.path.join(self.negative_path, name[0], name[1] + '.ply')
                    partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + f'_{i}.ply')
                    gt_paths.append(gt_ply_path)
                    negative_paths.append(negative_ply_path)
                    partial_paths.append(partial_ply_path)
            else:
                gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
                negative_ply_path = os.path.join(self.negative_path, name[0], name[1] + '.ply')
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '.ply')
                # if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
                gt_paths.append(gt_ply_path)
                negative_paths.append(negative_ply_path)
                partial_paths.append(partial_ply_path)

        return split_names, gt_paths, negative_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


class PCNConGrid(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, negative_path, n_pts, partial_pts):
        super(PCNConGrid, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.negative_path = os.path.join(data_root, split, negative_path)
        self.data_names, self.gt_paths, self.negative_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        if partial_pts is not None:
            self.partial_pts = partial_pts
        else:
            self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # read partial cloud
        partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_ply(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32)
        partial_enc = positional_encoding(partial_pc).transpose(1, 0)
        partial_pc = partial_pc.transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)
        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)

        if self.split == 'test':
            return {"id": "/".join(self.data_names[index]), "gt_points": pc,
                    "partial_id": 0, "partial_points": partial_pc, "partial_encoded": partial_enc}
        else:
            # read negative (not on surface) cloud
            negative_path = self.negative_paths[index]
            npc = read_point_cloud_ply(negative_path)
            # sample from negative cloud
            npc = random_sample(npc, self.n_pts)
            npc = torch.tensor(npc, dtype=torch.float32).transpose(1, 0)

            return {"id": "/".join(self.data_names[index]), "gt_points": pc, "negative_points": npc,
                    "partial_id": index % 8, "partial_points": partial_pc, "partial_encoded": partial_enc}

    def _load_data(self, category):
        split_dict = split_data_by_cat(self.data_root, category)
        split_names = split_dict[self.split]
        gt_paths = list()
        negative_paths = list()
        partial_paths = list()
        for name in split_names:
            if self.split == 'train':
                for i in range(8):
                    gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
                    negative_ply_path = os.path.join(self.negative_path, name[0], name[1] + '.ply')
                    partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + f'_{i}.ply')
                    gt_paths.append(gt_ply_path)
                    negative_paths.append(negative_ply_path)
                    partial_paths.append(partial_ply_path)
            else:
                gt_ply_path = os.path.join(self.gt_path, name[0], name[1] + '.ply')
                negative_ply_path = os.path.join(self.negative_path, name[0], name[1] + '.ply')
                partial_ply_path = os.path.join(self.partial_path, name[0], name[1] + '.ply')
                # if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
                gt_paths.append(gt_ply_path)
                negative_paths.append(negative_ply_path)
                partial_paths.append(partial_ply_path)

        return split_names, gt_paths, negative_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


class PCNHess(Dataset):
    def __init__(self, split, data_root, category, gt_path, partial_path, n_pts, partial_pts):
        super(PCNHess, self).__init__()
        self.split = split
        # self.shuffle = (split == "train")
        self.rng = random.Random(1234)
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, split, gt_path)
        self.partial_path = os.path.join(data_root, split, partial_path)
        self.data_names, self.gt_paths, self.partial_paths = self._load_data(category)
        self.n_pts = n_pts
        if partial_pts is not None:
            self.partial_pts = partial_pts
        else:
            self.partial_pts = n_pts // 2

    def __getitem__(self, index):
        # grid extra buffer
        eps = 0.1
        # read partial cloud
        render_choice = 0
        if self.split == 'train':
            render_choice = self.rng.randint(0, 7)
            partial_path = self.partial_paths[index].format(render_choice)
        else:
            partial_path = self.partial_paths[index]
        partial_pc = read_point_cloud_ply(partial_path)
        # modify partial cloud
        partial_pc = add_noise_pc(partial_pc)
        partial_pc = random_sample(partial_pc, self.partial_pts)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float32)
        partial_enc = positional_encoding(partial_pc).transpose(1, 0)
        partial_pc = partial_pc.transpose(1, 0)

        # read ground truth cloud
        gt_path = self.gt_paths[index]
        pc = read_point_cloud_ply(gt_path)
        # sample from ground truth cloud
        pc = random_sample(pc, self.n_pts)

        # find the bounding box for all dataset
        box_min = np.amin(pc, 0) - eps
        box_max = np.amax(pc, 0) + eps

        # points not on surface
        non_manifold_pts = np.random.uniform(box_min, box_max, size=(self.n_pts, 3)).astype(np.float32)
        non_manifold_pts = torch.from_numpy(non_manifold_pts).float()

        pc = torch.tensor(pc, dtype=torch.float32)

        # points close to surface
        dist = torch.cdist(pc, pc)
        sigmas = torch.topk(dist, k=51, dim=1, largest=False)[0][:, -1:]  # (n_points, 1)
        near_pts = (pc + sigmas * torch.randn(pc.shape[0], pc.shape[1]))

        pc_enc = positional_encoding(pc)  # .transpose(1, 0)
        # pc = pc.transpose(1, 0)

        return {"id": "/".join(self.data_names[index]), "gt_points": pc, "gt_encoded": pc_enc,
                "partial_id": render_choice, "partial_points": partial_pc, "partial_encoded": partial_enc,
                "non_manifold_points": non_manifold_pts, "near_points": near_pts}

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
            # if os.path.exists(gt_ply_path) and os.path.exists(partial_ply_path):
            gt_paths.append(gt_ply_path)
            partial_paths.append(partial_ply_path)

        return split_names, gt_paths, partial_paths

    def __len__(self):
        return len(self.data_names)


if __name__ == "__main__":
    pass
