import torch
import pytorch_lightning as pl

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

from models import get_model

import os
from tqdm import tqdm


class INR(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.model = get_model(config, "HessSimple")


def test_hessian_simple():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    ckpt_path = os.path.join(config.model_dir, 'model.ckpt')
    INR(config)
    model = INR.load_from_checkpoint(ckpt_path)
    print(model)


if __name__ == '__main__':
    test_hessian_simple()
