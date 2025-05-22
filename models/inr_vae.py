import torch
import torch.nn as nn
from dciknn_cuda import DCI
from models import gen_nearest_latents_with_indices


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
		self.n_features = list(n_features) + [2*latent_dim]
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
		mean, log_var = torch.split(x, self.latent_dim, -1)
		return mean, log_var


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
	def __init__(self, n_features=(256, 256), latent_dim=256, noise_dim=32, normalize=False, space_dim=3):
		super(DecoderFC, self).__init__()
		self.n_features = list(n_features) + [1]
		self.latent_dim = latent_dim
		self.noise_dim = noise_dim
		self.space_dim = space_dim

		model = []
		prev_nf = latent_dim + noise_dim + space_dim
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
		return x


class ImplicitVAE(nn.Module):
	def __init__(self, config):
		super(ImplicitVAE, self).__init__()
		self.encoder = EncoderPC(config.enc_features_inr, config.latent_dim, config.res_layers, config.enc_norm,
								config.space_dim)
		self.decoder = DecoderFC(config.dec_features_inr, config.latent_dim, config.noise_dim_inr, config.dec_norm,
								config.space_dim)
		self.dci_db = DCI(config.latent_dim, 2, 10, 100, 10)
		self.zeros = torch.zeros((config.batch_size, config.n_pts))

	def encode(self, x):
		return self.encoder(x)

	@staticmethod
	def reparameterization(mean, var):
		epsilon = torch.randn_like(var)
		z = mean + var * epsilon
		return z

	def decode(self, x):
		return self.decoder(x)

	def forward(self, partial_pts, manifold_pts, non_manifold_pts, near_pts, noises):
		mean, log_var = self.encoder(partial_pts)
		z = self.reparameterization(mean, torch.exp(0.5 * log_var))
		multi_z = z.unsqueeze(1).repeat(1, manifold_pts.size(-1), 1)
		manifold_pred_list = []
		for noise in noises:
			multi_noise = noise.unsqueeze(1).repeat(1, manifold_pts.size(-1), 1)
			manifold_pts_pred = self.decoder(torch.cat([multi_z, multi_noise, manifold_pts], dim=-1))
			manifold_pred_list.append(manifold_pts_pred)

		non_manifold_pts_pred = self.decoder(torch.cat([multi_z, multi_noise, non_manifold_pts], dim=-1))
		near_pts_pred = self.decoder(torch.cat([multi_z, multi_noise, near_pts], dim=-1))

		return {"manifold_pts_pred": manifold_pts_pred,
				"non_manifold_pts_pred": non_manifold_pts_pred,
				'near_pts_pred': near_pts_pred,
				"latent_mean": mean,
				"latent_log_var": log_var}


if __name__ == '__main__':
	pass
