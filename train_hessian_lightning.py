import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from models import get_model
from metrics import hessMorse

import os
import torch

from configs import get_config
from datasets import get_dataloader

import wandb

wandb.login(key='808f5ae6de6f014806ca1c9b374cdbae14787c5e')
wandb.init()

wandb_logger = WandbLogger(log_model='all')

args = get_config('train')
wandb.config.update(args)

# create dataloader
train_loader = get_dataloader('train', args)
val_loader = get_dataloader('validation', args)


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.args = config

    @staticmethod
    def train_dataloader():
        return train_loader


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.args = config
        self.learning_rate = config.lr
        if config.simple_hessian:
            self.net = get_model(config, "HessSimple")
        else:
            self.net = get_model(config, "HessComplex")

        self.criterion = hessMorse(weights=config.loss_weights, div_decay=config.morse_decay,
                                   div_type=config.morse_type, bidirectional_morse=config.bidirectional_morse)

    def training_step(self, data):
        self.net.train()
        self.net.zero_grad(set_to_none=True)
        partial_pc, partial_enc, manifold_pts, non_manifold_pts, near_pts = (
            data['partial_points'], data['partial_encoded'], data['gt_points'], data['non_manifold_points'],
            data['near_points'])
        manifold_pts.requires_grad_()
        non_manifold_pts.requires_grad_()
        near_pts.requires_grad_()

        output_pred = self.net(partial_enc, manifold_pts, non_manifold_pts, near_pts)

        loss, loss_dict, _ = self.criterion(output_pred, manifold_pts, non_manifold_pts, near_pts)
        """wandb_logger.log_text(f'Epoch: {self.current_epoch}, Loss: {loss_dict["loss"]}, L_Manifold: {loss_dict["sdf_term_manifold"]},' +
                             f'L_NonManifold: {loss_dict["sdf_term_non_manifold"]}, L_Eikonal: {loss_dict["eikonal_term"]},' +
                             f', L_Morse: {loss_dict["hessian_term"]} + L_Latent: {loss_dict["latent_reg_term"]}')"""
        log_dict = loss_dict
        log_dict['epoch'] = self.current_epoch
        wandb.log(log_dict)
        return {'loss': loss, 'manifold': manifold_pts[:1]}

    def on_training_epoch_end(self):
        self.net.eval()
        # update weights
        curr_epoch = self.current_epoch
        self.criterion.update_morse_weight(curr_epoch, self.args.num_epochs, self.args.decay_params)

    def configure_optimizers(self):
        # Setup Adam optimizers
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.learning_rate, amsgrad=True)
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150 * 10, T_mult=2, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sch,
                "interval": "step",
                "frequency": 1,
            },
        }


check_callback = ModelCheckpoint(
    dirpath=args.model_dir,
    filename='model-{epoch:02d}',
    save_top_k=2,
    save_last=True
)
lr_monitor = LearningRateMonitor(logging_interval='step')
pl.seed_everything(args.seed, workers=True)
trainer = pl.Trainer(gradient_clip_val=args.grad_clip_norm,
                     max_epochs=args.num_epochs,
                     accelerator='gpu',
                     strategy=DDPStrategy(find_unused_parameters=False),
                     devices=1,
                     callbacks=[check_callback, TQDMProgressBar(refresh_rate=10), lr_monitor],
                     accumulate_grad_batches=8,
                     benchmark=True,
                     deterministic=False,
                     logger=wandb_logger,
                     )
base_trainer = BaseTrainer(args)
dm = DataModule(args)

trainer.fit(base_trainer, dm, ckpt_path=os.path.join(args.model_dir, 'last.ckpt') if os.path.exists(
    os.path.join(args.model_dir, 'last.ckpt')) else None)

wandb.finish()
