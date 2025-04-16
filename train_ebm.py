from skimage.morphology import reconstruction
from sympy.physics.quantum.density import fidelity

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

from tqdm import tqdm
from collections import OrderedDict

import wandb


def train_ebm():
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
    wandb.watch(trainer.model)

    for e in range(watcher.epoch, config.num_epochs):
        # begin iteration
        total_loss = 0.0
        reconstruction_loss = 0.0
        fidelity_loss = 0.0
        latent_mse_loss = 0.0
        encoder_decoder_loss = 0.0
        ebm_loss = 0.0
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            trainer.train_func(data)

            # visualize
            if config.vis and watcher.step % config.vis_frequency == 0:
                trainer.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            losses = trainer.collect_loss()
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
            reconstruction_loss += losses['gt_recon']
            fidelity_loss += losses['part_fidelity']
            latent_mse_loss += losses['imle']
            encoder_decoder_loss += losses['coder_loss']
            ebm_loss += losses['energy_loss']
            for _, v in losses.items():
                total_loss += v.item()

            # validation step
            if watcher.step % config.val_frequency == 0:
                data = next(val_loader)
                trainer.val_func(data)

                if config.vis and watcher.step % config.vis_frequency == 0:
                    trainer.visualize_batch(data, "validation")

            watcher.within_epoch()

        trainer.update_learning_rate()
        watcher.new_epoch()
        ebm_loss /= len(pbar)
        reconstruction_loss /= len(pbar)
        fidelity_loss /= len(pbar)
        latent_mse_loss /= len(pbar)
        encoder_decoder_loss /= len(pbar)
        total_loss /= len(pbar)
        log_dict = {
            'epoch': e,
            'ebm_loss': ebm_loss,
            'gt_reconstruction_loss': reconstruction_loss,
            'partial_fidelity_loss': fidelity_loss,
            'latent_mse': latent_mse_loss,
            'encoder_decoder_total_loss': encoder_decoder_loss,
            'total_loss': total_loss
        }
        wandb.log(log_dict)

        if watcher.epoch % config.save_frequency == 0:
            trainer.save_ckpt()
        trainer.save_ckpt('latest')


if __name__ == '__main__':
    train_ebm()
