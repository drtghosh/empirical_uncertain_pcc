import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

import os
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

from metrics import ldf
from metrics.EMD import emd

import matplotlib.pyplot as plt


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

	# initialize emd instance
	criterion = emd()
	# test
	# things to store for mean
	all_emds = torch.zeros(config.gen_samples_test * num_test)
	naive_mean_emds = torch.zeros(num_test)
	matched_mean_emds = torch.zeros(num_test)
	all_udhs = torch.zeros(config.gen_samples_test * num_test)
	naive_mean_uhds = torch.zeros(num_test)
	matched_mean_udhs = torch.zeros(num_test)

	# things to store for std
	naive_all_std_norms = torch.zeros(config.n_pts * num_test)
	naive_max_std_norms = torch.zeros(num_test)
	naive_avg_std_norms = torch.zeros(num_test)
	naive_min_std_norms = torch.zeros(num_test)
	matched_all_std_norms = torch.zeros(config.n_pts * num_test)
	matched_max_std_norms = torch.zeros(num_test)
	matched_avg_std_norms = torch.zeros(num_test)
	matched_min_std_norms = torch.zeros(num_test)
	# loop
	for it in tqdm(range(num_test)):
		data = next(test_loader)
		with torch.no_grad():
			trainer.forward(data, False)
		num_gen = len(trainer.latent_gen_list)
		gen_clouds = torch.zeros(num_gen, trainer.complete_pc.size(-1), trainer.complete_pc.size(1))
		for j in range(num_gen):
			latent = trainer.latent_gen_list[j]
			gen_pc_tensor = trainer.pointAE.decode(latent)
			gen_clouds[j] = gen_pc_tensor[0].transpose(1, 0)
			all_udhs[it*config.gen_samples_test + j] = ldf(trainer.partial_pc, gen_pc_tensor)
			emd_dis, _ = criterion(gen_pc_tensor.transpose(1, 2), trainer.complete_pc.transpose(1, 2), 0.05, 3000)
			all_emds[it*config.gen_samples_test + j] = torch.mean(torch.sqrt(emd_dis))

		# naive estimation
		# mean
		gen_mu_naive = gen_clouds.mean(dim=0)
		naive_mean_uhds[it] = ldf(trainer.partial_pc, gen_mu_naive.transpose(1, 0).unsqueeze(0))
		emd_dis_naive, _ = criterion(gen_mu_naive.unsqueeze(0), trainer.complete_pc.transpose(1, 2), 0.05, 3000)
		naive_mean_emds[it] = torch.mean(torch.sqrt(emd_dis_naive))
		# std
		gen_std_naive = gen_clouds.std(dim=0)
		std_norm_naive = torch.norm(gen_std_naive, dim=[1])
		naive_all_std_norms[it * config.n_pts:(it+1) * config.n_pts] = std_norm_naive
		std_max_naive = std_norm_naive.max()
		naive_max_std_norms[it] = std_max_naive
		std_avg_naive = std_norm_naive.mean()
		naive_avg_std_norms[it] = std_avg_naive
		std_min_naive = std_norm_naive.min()
		naive_min_std_norms[it] = std_min_naive

		# linear assignment estimation
		for i in range(1, len(gen_clouds)):
			cost_matrix = torch.cdist(gen_clouds[0], gen_clouds[i], p=2)
			_, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
			gen_clouds[i] = gen_clouds[i][col_ind, :]
		# mean
		gen_mu_matched = gen_clouds.mean(dim=0)
		matched_mean_udhs[it] = ldf(trainer.partial_pc, gen_mu_matched.transpose(1, 0).unsqueeze(0))
		emd_dis_matched, _ = criterion(gen_mu_matched.unsqueeze(0), trainer.complete_pc.transpose(1, 2), 0.05, 3000)
		matched_mean_emds[it] = torch.mean(torch.sqrt(emd_dis_matched))
		# std
		gen_std_matched = gen_clouds.std(dim=0)
		std_norm_matched = torch.norm(gen_std_matched, dim=[1])
		matched_all_std_norms[it * config.n_pts:(it + 1) * config.n_pts] = std_norm_matched
		std_max_matched = std_norm_matched.max()
		matched_max_std_norms[it] = std_max_matched
		std_avg_matched = std_norm_matched.mean()
		matched_avg_std_norms[it] = std_avg_matched
		std_min_matched = std_norm_matched.min()
		matched_min_std_norms[it] = std_min_matched

		# plot emds
		plt.hist(all_emds.numpy())
		plt.show()


if __name__ == '__main__':
	test_imle_gen_metrics()
