from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

from tqdm import tqdm
from collections import OrderedDict

import wandb


def train_hessian():
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
		manifold_loss = 0.0
		non_manifold_loss = 0.0
		eikonal_loss = 0.0
		hessian_loss = 0.0
		latent_reg_loss = 0.0
		pbar = tqdm(train_loader)
		for b, data in enumerate(pbar):
			# train step
			trainer.train_func(data)

			pbar.set_description("EPOCH[{}][{}]".format(e, b))
			losses = trainer.collect_loss()
			pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
			total_loss += losses['loss']
			manifold_loss += losses['sdf_term_manifold']
			non_manifold_loss += losses['sdf_term_non_manifold']
			eikonal_loss += losses['eikonal_term']
			hessian_loss += losses['hessian_term']
			latent_reg_loss += losses['latent_reg_term']

			# validation step
			if watcher.step % config.val_frequency == 0:
				data = next(val_loader)
				trainer.val_func(data)

			watcher.within_epoch()

		trainer.update_learning_rate()
		watcher.new_epoch()
		total_loss /= len(pbar)
		manifold_loss /= len(pbar)
		non_manifold_loss /= len(pbar)
		eikonal_loss /= len(pbar)
		hessian_loss /= len(pbar)
		latent_reg_loss /= len(pbar)
		log_dict = {
			'epoch': e,
			'total_loss': total_loss,
			'manifold_loss': manifold_loss,
			'non_manifold_loss': non_manifold_loss,
			'eikonal_loss': eikonal_loss,
			'hessian_loss': hessian_loss,
			'latent_reg_loss': latent_reg_loss
		}
		wandb.log(log_dict)

		if watcher.epoch % config.save_frequency == 0:
			trainer.save_ckpt()
		trainer.save_ckpt('latest')


if __name__ == '__main__':
	train_hessian()
