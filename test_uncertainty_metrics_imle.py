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
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def plot_distances(distances, n_bins=20):
	fig, axs = plt.subplots(1, 1, tight_layout=True)
	# N is the count in each bin, bins is the lower-limit of the bin
	N, bins, patches = axs.hist(distances.cpu().numpy(), bins=n_bins, density=True)

	# We'll color code by height, but you could use any scalar
	fracs = N / N.max()

	# we need to normalize the data to 0..1 for the full range of the colormap
	norm = colors.Normalize(fracs.min(), fracs.max())

	# Now, we'll loop through our objects and set the color of each accordingly
	for thisfrac, thispatch in zip(fracs, patches):
		color = plt.cm.viridis(norm(thisfrac))
		thispatch.set_facecolor(color)
	axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
	plt.show()


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
	all_emds = torch.zeros(config.gen_samples_test * num_test).to(config.device)
	naive_mean_emds = torch.zeros(num_test).to(config.device)
	matched_mean_emds = torch.zeros(num_test).to(config.device)
	all_udhs = torch.zeros(config.gen_samples_test * num_test).to(config.device)
	naive_mean_uhds = torch.zeros(num_test).to(config.device)
	matched_mean_udhs = torch.zeros(num_test).to(config.device)

	# things to store for std
	naive_all_std_norms = torch.zeros(config.n_pts * num_test).to(config.device)
	naive_max_std_norms = torch.zeros(num_test).to(config.device)
	naive_avg_std_norms = torch.zeros(num_test).to(config.device)
	naive_min_std_norms = torch.zeros(num_test).to(config.device)
	matched_all_std_norms = torch.zeros(config.n_pts * num_test).to(config.device)
	matched_max_std_norms = torch.zeros(num_test).to(config.device)
	matched_avg_std_norms = torch.zeros(num_test).to(config.device)
	matched_min_std_norms = torch.zeros(num_test).to(config.device)
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
		gen_mu_naive = gen_clouds.mean(dim=0).to(config.device)
		naive_mean_uhds[it] = ldf(trainer.partial_pc, gen_mu_naive.transpose(1, 0).unsqueeze(0))
		emd_dis_naive, _ = criterion(gen_mu_naive.unsqueeze(0), trainer.complete_pc.transpose(1, 2), 0.05, 3000)
		naive_mean_emds[it] = torch.mean(torch.sqrt(emd_dis_naive))
		# std
		gen_std_naive = gen_clouds.std(dim=0).to(config.device)
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
		gen_mu_matched = gen_clouds.mean(dim=0).to(config.device)
		matched_mean_udhs[it] = ldf(trainer.partial_pc, gen_mu_matched.transpose(1, 0).unsqueeze(0))
		emd_dis_matched, _ = criterion(gen_mu_matched.unsqueeze(0), trainer.complete_pc.transpose(1, 2), 0.05, 3000)
		matched_mean_emds[it] = torch.mean(torch.sqrt(emd_dis_matched))
		# std
		gen_std_matched = gen_clouds.std(dim=0).to(config.device)
		std_norm_matched = torch.norm(gen_std_matched, dim=[1])
		matched_all_std_norms[it * config.n_pts:(it + 1) * config.n_pts] = std_norm_matched
		std_max_matched = std_norm_matched.max()
		matched_max_std_norms[it] = std_max_matched
		std_avg_matched = std_norm_matched.mean()
		matched_avg_std_norms[it] = std_avg_matched
		std_min_matched = std_norm_matched.min()
		matched_min_std_norms[it] = std_min_matched

		# plot emds
		plot_distances(all_emds)


if __name__ == '__main__':
	test_imle_gen_metrics()
