import torch
import torch.nn as nn


def weights_init(m):
	class_name = m.__class__.__name__
	if class_name.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif class_name.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


class Generator(nn.Module):
	def __init__(self, n_features=(256, 512), latent_dim=128, noise_dim=8, normalize=False):
		super(Generator, self).__init__()
		self.n_features = list(n_features) + [latent_dim]
		self.latent_dim = latent_dim
		self.noise_dim = noise_dim

		model = []
		prev_nf = latent_dim + noise_dim
		for idx, nf in enumerate(self.n_features):
			fc_layer = nn.Linear(prev_nf, nf)
			model.append(fc_layer)

			if normalize:
				norm_layer = nn.BatchNorm1d(nf)
				model.append(norm_layer)

			if idx < len(n_features):
				activation_layer = nn.LeakyReLU(inplace=True)
				model.append(activation_layer)

			prev_nf = nf

		self.model = nn.Sequential(*model)

		self.apply(weights_init)

	def forward(self, x, noise):
		x = torch.cat([x, noise], dim=1)
		x = self.model(x)
		return x


if __name__ == '__main__':
	pass
