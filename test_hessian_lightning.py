import torch
import numpy as np
import pytorch_lightning as pl

from configs import get_config

from models import get_model
from datasets.data_utils import read_point_cloud_ply, positional_encoding

import os
from tqdm import tqdm


def test_hessian_simple():
    # create experiment config containing all hyperparameters
    config = get_config('test')
    config.latent_dim = 256
    eps = 0.25

    ckpt_path = os.path.join(config.model_dir, 'model.ckpt')
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model = get_model(config, "HessSimple")
    model.load_state_dict(new_state_dict)
    pc = read_point_cloud_ply(os.path.join(config.result_dir, 'partial.ply'))
    pc = torch.tensor(pc, dtype=torch.float32)
    enc_pc = positional_encoding(pc).transpose(1, 0).unsqueeze(0)
    enc = model.encoder(enc_pc)
    # find the bounding box for all dataset
    box_min = np.amin(pc, 0) - eps
    box_max = np.amax(pc, 0) + eps

    grid_sizes = np.ones(3, dtype=np.int32) * 128
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(3)])
    grid_vertices = torch.from_numpy(grid_vertices).float().unsqueeze(0)


if __name__ == '__main__':
    test_hessian_simple()
