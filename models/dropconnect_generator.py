import torch.nn as nn
from .model_utils import WeightDrop


def weights_init(m):
	class_name = m.__class__.__name__
	if class_name.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif class_name.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class GeneratorDropConnect(nn.Module):
	def __init__(self, config):
		super(GeneratorDropConnect, self).__init__()
		self.n_features = list(config.n_features_gen) + [config.latent_dim]
		self.latent_dim = config.latent_dim
		self.noise_dim = config.noise_dim
		self.dropout_prob = config.dropout_prob
		self.device = config.device

		model = []
		prev_nf = config.latent_dim
		for idx, nf in enumerate(self.n_features):
			if 0 < idx < len(config.n_features_gen):
				fc_layer = WeightDrop(nn.Linear(prev_nf, nf), ['weight'], self.device, self.dropout_prob)
				print(fc_layer.device)
			else:
				fc_layer = nn.Linear(prev_nf, nf).to(self.device)
				print(fc_layer.device)
			model.append(fc_layer)

			if config.gen_norm:
				norm_layer = nn.BatchNorm1d(nf)
				model.append(norm_layer)

			if idx < len(config.n_features_gen):
				activation_layer = nn.LeakyReLU(inplace=True)
				model.append(activation_layer)

			prev_nf = nf

		self.model = nn.Sequential(*model)

		self.apply(weights_init)

	def forward(self, x):
		x = self.model(x)
		return x


if __name__ == '__main__':
	pass
