import torch
import numpy as np
import pytorch_lightning as pl

from configs import get_config

from models import get_model
from datasets.data_utils import read_point_cloud_ply, positional_encoding

from torch.distributions import normal

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
    z = model.encoder(enc_pc)
    z_sampler = normal.Normal(0, 1)
    noise = z_sampler.sample([pc.size(0), config.noise_dim_inr]).to(config.device)
    # find the bounding box for all dataset
    box_min = torch.amin(pc, 0) - eps
    print(box_min)
    box_max = torch.amax(pc, 0) + eps

    grid_sizes = np.ones(3, dtype=np.int32) * 128
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(3)])
    multi_z = z.unsqueeze(1).repeat(1, len(grid_vertices), 1)
    multi_noise = noise.unsqueeze(1).repeat(1, len(grid_vertices), 1).to(config.device)
    grid_vertices = torch.from_numpy(grid_vertices).float().unsqueeze(0)
    pred = model.decoder(torch.cat([grid_vertices, multi_z, multi_noise], dim=-1)).squeeze(-1)
    print(pred)
    print(pred.shape)


if __name__ == '__main__':
    test_hessian_simple()
