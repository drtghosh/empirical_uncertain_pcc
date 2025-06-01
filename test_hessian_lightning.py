import torch

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
    ckpt_dict = torch.load(ckpt_path)
    model = get_model(config, "HessSimple")
    model._load_from_state_dict(ckpt_dict['state_dict'])
    print(model)


if __name__ == '__main__':
    test_hessian_simple()
