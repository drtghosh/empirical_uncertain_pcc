import torch
import torch.nn as nn
import numpy as np


class EncoderPC(nn.Module):
	"""
		Encoder class for encoding a 2/3D point cloud.
		Args-
			n_features: tuple of number of features (#filters used in 1D convolutions) in each forward layer
			latent_dim: latent dimension of the point cloud encoding representation
			residual_layers: tuple of layer indices where earlier global feature is concatenated to local features
			normalize: boolean indicating batch normalization is used
			space_dim: dimension of the data space
	"""

	def __init__(self, n_features=(64, 128, 256, 256), latent_dim=256, residual_layers=(2,), normalize=True, space_dim=3):
		super(EncoderPC, self).__init__()
		self.n_features = list(n_features) + [latent_dim]
		self.latent_dim = latent_dim
		self.residual_layers = list(residual_layers)
		self.normalize = normalize
		assert self.residual_layers[-1] < len(n_features), "global feature should not be concatenated in the final layer"

		model = []
		# fix positional encoding size here
		pos_enc = 6
		prev_nf = space_dim + (pos_enc * pos_enc)
		for idx, nf in enumerate(self.n_features):
			if idx in self.residual_layers:
				prev_nf = 2 * prev_nf
			conv_layer = nn.Conv1d(prev_nf, nf, 1)
			model.append(conv_layer)

			if normalize:
				norm_layer = nn.BatchNorm1d(nf)
				model.append(norm_layer)

			activation_layer = nn.LeakyReLU(inplace=True)
			model.append(activation_layer)
			prev_nf = nf

		self.model = nn.Sequential(*model)

	def forward(self, pc):
		batch_size, _, partial_num = pc.shape  # B, _, N = batch_size, _, partial_num
		x = pc  # pc.transpose(2, 1)
		for idx, nf in enumerate(self.n_features):
			if idx in self.residual_layers:
				global_feature = torch.max(x, dim=2, keepdim=True)[0]
				x = torch.cat([global_feature.expand(-1, -1, partial_num), x], dim=1)
			if self.normalize:
				x = self.model[3*idx + 2](self.model[3*idx + 1](self.model[3*idx](x)))
			else:
				x = self.model[2 * idx + 1](self.model[2 * idx](x))
		x = torch.max(x, dim=2)[0]
		return x


class DecoderFC(nn.Module):
	"""
		Decoder class for reconstructing a 2/3D point cloud from the encoding.
		Args-
			n_features: tuple of number of features (#filters used in 1D convolutions) in each forward layer
			latent_dim: latent dimension of the point cloud encoding representation
			output_pts: number of points to output per cloud
			normalize: boolean indicating batch normalization is used
			space_dim: dimension of the data space
	"""
	def __init__(self, n_features=(256, 256), latent_dim=256, output_pts=2048, normalize=False, space_dim=3):
		super(DecoderFC, self).__init__()
		self.n_features = list(n_features) + [output_pts * space_dim]
		self.output_pts = output_pts
		self.latent_dim = latent_dim
		self.space_dim = space_dim

		model = []
		prev_nf = self.latent_dim
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

	def forward(self, x):
		x = self.model(x)
		x = x.view((-1, self.space_dim, self.output_pts))
		return x


class EBM(nn.Module):
	def __init__(self, step_size=1, n_step=32, noise_scale=None):
		super(EBM, self).__init__()
		self.step_size = step_size
		self.n_step = n_step
		self.noise_scale = noise_scale
		if noise_scale is None:
			self.noise_scale = np.sqrt(step_size * 2)


class EBMCompletion(nn.Module):
	def __init__(self, config):
		super(EBMCompletion, self).__init__()
		self.encoder = EncoderPC(config.enc_features, config.latent_dim, config.res_layers, config.enc_norm, config.space_dim)
		self.decoder = DecoderFC(config.dec_features, config.latent_dim, config.n_pts, config.dec_norm, config.space_dim)

	def encode(self, x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x, is_partial=True):
		z = self.encoder(x)
		x = self.decoder(z)
		return x


if __name__ == '__main__':
	pass
