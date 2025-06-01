import torch
import pytorch_lightning as pl

from configs import get_config

from models import get_model
from datasets.data_utils import read_point_cloud_ply

import os
from tqdm import tqdm


def test_hessian_simple(filepath):
    # create experiment config containing all hyperparameters
    config = get_config('test')
    config.latent_dim = 256

    ckpt_path = os.path.join(config.model_dir, 'model.ckpt')
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model = get_model(config, "HessSimple")
    model.load_state_dict(new_state_dict)
    pc = read_point_cloud_ply(os.path.join(config.result_dir, 'partial.ply'))
    pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0).unsqueeze(0)
    enc = model.encoder(pc)
    print(enc)
    print(enc.shape)


if __name__ == '__main__':
    test_hessian_simple()
