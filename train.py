from configs import get_config
from collections import OrderedDict
from tools import get_trainer


def train_model():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # create model and trainer
    trainer = get_trainer(config)
    # load from checkpoint if provided
    if config.cont:
        trainer.load_ckpt(config.ckpt)
        print('Model Loaded from checkpoint, continue training...')
