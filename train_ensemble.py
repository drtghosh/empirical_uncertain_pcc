import numpy as np

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

from tqdm import tqdm
from collections import OrderedDict

import wandb


def train_ensemble():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # weights and biases
    wandb.login(key='808f5ae6de6f014806ca1c9b374cdbae14787c5e')
    wandb.init()
    wandb.config.update(config)

    # create model and trainer
    trainer = get_trainer(config)
    # load from checkpoint if provided
    if config.cont:
        trainer.load_ckpt(config.ckpt)
        print('Model Loaded from checkpoint, continue training...')

    # create dataloader
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('validation', config)
    val_loader = cycle(val_loader)

    # start training watcher
    watcher = trainer.watcher

    # weights and biases watch
    for model in trainer.models:
        wandb.watch(model)

    for e in range(watcher.epoch, config.num_epochs):
        # begin iteration
        total_losses = np.zeros(trainer.n_models)
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            trainer.train_func(data)

            # visualize
            if config.vis and watcher.step % config.vis_frequency == 0:
                trainer.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            losses = trainer.collect_losses()
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
            for _, v in losses.items():
                for i in range(trainer.n_models):
                    total_losses[i] += v[i].item()

            # validation step
            if watcher.step % config.val_frequency == 0:
                data = next(val_loader)
                trainer.val_func(data)

                if config.vis and watcher.step % config.vis_frequency == 0:
                    trainer.visualize_batch(data, "validation")

            watcher.within_epoch()

        trainer.update_learning_rate()
        watcher.new_epoch()
        total_losses /= len(pbar)
        loss_log_dict = dict({'epoch': e})
        for i in range(trainer.n_models):
            loss_log_dict[f'model_{i}_loss'] = total_losses[i]
        wandb.log(loss_log_dict)

        if watcher.epoch % config.save_frequency == 0:
            trainer.save_ckpt()
        trainer.save_ckpt('latest')


if __name__ == '__main__':
    train_ensemble()
