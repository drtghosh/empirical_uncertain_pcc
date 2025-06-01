import torch
import numpy as np

from configs import get_config

from models import get_model
from datasets.data_utils import read_point_cloud_ply, positional_encoding

from torch.distributions import normal

import os


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
    model.to(config.device)
    pc = read_point_cloud_ply(os.path.join(config.result_dir, 'partial.ply'))
    pc = torch.tensor(pc, dtype=torch.float32)
    enc_pc = positional_encoding(pc).transpose(1, 0).unsqueeze(0).to(config.device)
    z = model.encoder(enc_pc).to(config.device)
    z_sampler = normal.Normal(0, 1)
    noise = z_sampler.sample([1, config.noise_dim_inr]).to(config.device)
    # find the bounding box for all dataset
    box_min = torch.amin(pc, 0) - eps
    box_max = torch.amax(pc, 0) + eps

    grid_sizes = np.ones(3, dtype=np.int32) * 64
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(3)])
    grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, 3)
    grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
    loop = len(grid_vertices) // (128*8)
    multi_z = z.unsqueeze(1).repeat(1, loop, 1)
    multi_noise = noise.unsqueeze(1).repeat(1, loop, 1).to(config.device)
    grid_vertices = grid_vertices.unsqueeze(0).to(config.device)
    all_predictions = []
    for i in range(128*8):
        pred = model.decoder(torch.cat([grid_vertices[:, i*loop:(i+1)*loop, :], multi_z, multi_noise], dim=-1)).squeeze()
        all_predictions.append(pred)
    final_pred = torch.stack(all_predictions)
    print(final_pred.shape)
    print(final_pred)


if __name__ == '__main__':
    test_hessian_simple()
