import torch
from models import get_model, gen_nearest_latents
from tools.base_trainer import TrainerCommonEBM
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
		self.fidelity_loss = None
		self.reconstruction_loss = None
		self.recon_weight = config.recon_weight_ebm
		self.fidelity_weight = config.fidelity_weight_ebm
		self.latent_gen_weight = config.latent_gen_weight_ebm
		self.energy_reg_weight = config.regularization_weight_ebm
		if config.is_train:
			self.z_samples = config.gen_samples_train
		else:
			self.z_samples = config.gen_samples_test
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

		recon_complete = self.model(complete_enc, False)

		# to store the generated latent encodings for complete cloud
		self.latent_gen_list = []

		if train:
			# compute reconstruction loss
			emd_dis, assignment = self.criterionRecon(recon_complete.transpose(1, 2), self.complete_pc.transpose(1, 2),
													0.05, 3000)
			self.reconstruction_loss = self.recon_weight * torch.mean(torch.sqrt(emd_dis))

			# collect generated latent encodings
			for idx in range(self.z_samples):
				latent_gen = self.model.ebm.sample_langevin(partial_latent)
				self.latent_gen_list.append(latent_gen)
				latent_gen_list = torch.stack(self.latent_gen_list, 1)
				# sample closest to gt encoding
				latent_gen_nearest = gen_nearest_latents(self.dci_db, latent_gen_list, complete_latent)
				self.gen_pc = self.model.decode(latent_gen_nearest)

				# compute latent and fidelity loss
				self.latent_gen_loss = self.latent_gen_weight * self.criterionLatent(latent_gen_nearest,
																					complete_latent)
				self.fidelity_loss = self.fidelity_weight * ldf(self.partial_pc, self.gen_pc)
				# add up to get encoder decoder loss
				self.encoder_decoder_loss = self.reconstruction_loss + self.fidelity_loss + self.latent_gen_loss

				# compute ebm loss
				latent_energy = self.model.ebm.get_energy(latent_gen_nearest)
				gt_energy = self.model.ebm.get_energy(complete_latent)
				regularization_term = self.energy_reg_weight * (gt_energy ** 2 + latent_energy ** 2).mean()
				self.ebm_loss = (gt_energy.mean() - latent_energy.mean) + regularization_term
		else:
			for idx in range(self.z_samples):
				self.model.ebm.eval()
				with torch.no_grad():
					latent_gen = self.model.ebm.sample_langevin(partial_latent)
				self.latent_gen_list.append(latent_gen)

	def collect_loss(self):
		loss_dict = {
			"gt_recon": self.reconstruction_loss,
			"part_fidelity": self.fidelity_loss,
			"imle": self.latent_gen_loss,
			"coder_loss": self.encoder_decoder_loss,
			"energy_loss": self.ebm_loss
		}
		return loss_dict

	def update_models(self):
		# update encoder decoder
		self.optimizer_ed.zero_grad()
		self.encoder_decoder_loss.backward()
		self.optimizer_ed.step()

		# update energy model
		self.optimizer_ebm.zero_grad()
		self.ebm_loss.backward()
		self.optimizer_ebm.step()

	def visualize_batch(self, data, mode, num=2, **kwargs):
		tbw = self.train_tbw if mode == 'train' else self.val_tbw

		target_pts = data['gt_points'][:num].transpose(1, 2).detach().cpu().numpy()
		partial_pts = data['partial_points'][:num].transpose(1, 2).detach().cpu().numpy()
		generated_pts = self.gen_pc[:num].transpose(1, 2).detach().cpu().numpy()

		# generated_pts = np.clip(generated_pts, -0.999, 0.999)

		tbw.add_mesh("gt", vertices=target_pts, global_step=self.watcher.step)
		tbw.add_mesh("partial", vertices=partial_pts, global_step=self.watcher.step)
		tbw.add_mesh("generated", vertices=generated_pts, global_step=self.watcher.step)
