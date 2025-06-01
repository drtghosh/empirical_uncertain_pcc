import torch
import pytorch_lightning as pl

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

from models import get_model

import os
from tqdm import tqdm


def test_hessian_simple():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    ckpt_path = os.path.join(config.model_dir, 'model.ckpt')
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model = get_model(config, "HessSimple")
    model.load_state_dict(new_state_dict)
    print(model)


if __name__ == '__main__':
    test_hessian_simple()
