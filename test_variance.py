import torch

from configs import get_config
from datasets import get_dataloader
from datasets.data_utils import cycle

import os
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

import pickle

import seaborn as sns


def plot_tensor(tensor, x_label=None, save_dir=None, filename=None):
	pt = sns.displot(tensor.cpu().numpy(), stat='percent', kde=True, kind='hist', element="step", bins=30)
	pt.set(xlabel=x_label)
	pt.figure.savefig(os.path.join(save_dir, filename + '.png'), bbox_inches="tight")
	with open(os.path.join(save_dir, filename), 'wb') as f:
		pickle.dump(tensor.cpu().numpy(), f)


def compute_data_variance():
	# create experiment config containing all hyperparameters
	config = get_config('test')
	# create dataloader
	config.batch_size = 1
	config.num_workers = 1
	test_loader = get_dataloader('test', config)
	num_test = len(test_loader)
	print(f"Total number of test samples: {num_test}.")
	test_loader = cycle(test_loader)

	# directory to save results
	save_dir = os.path.join("test_data_variance", f"{config.dataset_name}_{config.category}")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	all_data = torch.zeros((num_test, config.n_pts, 3)).to(config.device)
	for it in tqdm(range(num_test)):
		data = next(test_loader)
		all_data[it] = data['gt_points'].to(config.device)[0].transpose(0, 1)

	# linear assignment estimation
	for i in range(1, num_test):
		cost_matrix = torch.cdist(all_data[0], all_data[i], p=2)
		_, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
		all_data[i] = all_data[i][col_ind, :]
	# mean
	# mu_matched = all_data.mean(dim=0).to(config.device)
	# std
	std_matched = all_data.std(dim=0).to(config.device)
	std_norm_matched = torch.norm(std_matched, dim=[1])
	print(std_norm_matched.shape)
	plot_tensor(std_norm_matched, "Standard deviation norm", save_dir, 'std_norms')


if __name__ == '__main__':
	compute_data_variance()
