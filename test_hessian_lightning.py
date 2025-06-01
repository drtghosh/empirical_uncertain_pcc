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
    model = get_model(config, "HessSimple")
    model.load_state_dict(ckpt_path)
    print(model)


if __name__ == '__main__':
    test_hessian_simple()
