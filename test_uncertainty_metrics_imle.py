import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

import os
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment


def test_imle_gen_metrics():
	# create experiment config containing all hyperparameters
	config = get_config('test')

	# create model and trainer
	trainer = get_trainer(config)

	# load from checkpoint
	trainer.load_ckpt(config.ckpt)
	trainer.model.eval()

	# create dataloader
	config.batch_size = 1
	config.num_workers = 1
	test_loader = get_dataloader('test', config)
	num_test = len(test_loader)
	print(f"Total number of test samples: {num_test}.")
	test_loader = cycle(test_loader)

	# directory to save results
	save_dir = os.path.join(config.result_dir, f"ckpt-{config.ckpt}-z{config.gen_samples_test}-metrics")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# test
	for _ in tqdm(range(num_test)):
		data = next(test_loader)
		with torch.no_grad():
			trainer.forward(data, False)
		num_gen = len(trainer.latent_gen_list)
		gen_clouds = torch.zeros(num_gen, trainer.complete_pc.size(-1), trainer.complete_pc.size(1))
		for j in range(num_gen):
			latent = trainer.latent_gen_list[j]
			gen_pc_tensor = trainer.pointAE.decode(latent)[0].transpose(1, 0)
			gen_clouds[j] = gen_pc_tensor

		# naive estimation
		gen_mu_naive = gen_clouds.mean(dim=0)
		gen_std_naive = gen_clouds.std(dim=0)

		# linear assignment estimation
		for i in range(1, len(gen_clouds)):
			cost_matrix = torch.cdist(gen_clouds[0], gen_clouds[i], p=2)
			_, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
			gen_clouds[i] = gen_clouds[i][col_ind, :]
		gen_mu_matched = gen_clouds.mean(dim=0)
		gen_std_matched = gen_clouds.std(dim=0)
		print(gen_std_matched.max())
		print(gen_std_matched.min())
		break


if __name__ == '__main__':
	test_imle_gen_metrics()
