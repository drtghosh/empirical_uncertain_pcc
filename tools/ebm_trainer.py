import torch
from models import get_model, set_requires_grad, gen_nearest_latents
from tools.base_trainer import TrainerCommonEBM
from torch.distributions import normal
from dciknn_cuda import DCI
from metrics.EMD import emd
from metrics import ldf


class TrainerEBM(TrainerCommonEBM):
	def __init__(self, config):
		super(TrainerEBM, self).__init__(config)
		self.partial_pc = None
		self.complete_pc = None
		self.data_id = None
		self.latent_gen_loss = None
		self.reconstruction_loss = None
		self.recon_weight = config.recon_weight
		self.latent_gen_weight = config.latent_gen_weight
		self.z_dim = config.noise_dim
		if config.is_train:
			self.z_samples = config.gen_samples_train
		else:
			self.z_samples = config.gen_samples_test
		self.z_sampler = normal.Normal(0, 1)
		self.latent_gen_list = []
		# dci_db = DCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size, devices=[0, 1])
		self.dci_db = DCI(config.latent_dim, 2, 10, 100, 10)
		self.gen_pc = None

	def build_model(self, config):
		# customize the build_model function to build energy-based generative model
		model = get_model(config, "EBMgen").to(self.device)
		return model

	def set_loss_function(self):
		self.criterionLatent = self.criterionMSE
		self.criterionRecon = emd()

	def forward(self, data, train=True):
		self.data_id = data['id']
		self.partial_pc = data['partial_points'].to(self.device)
		partial_enc = data['partial_encoded'].to(self.device)
		self.complete_pc = data['gt_points'].to(self.device)
		complete_enc = data['gt_encoded'].to(self.device)

		partial_latent = self.model.encode(partial_enc)
		complete_latent = self.model.encode(complete_enc)

		self.latent_gen_list = []
		for idx in range(self.z_samples):


