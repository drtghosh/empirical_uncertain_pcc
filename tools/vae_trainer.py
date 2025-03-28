import torch
from models import get_model
from tools.base_trainer import TrainerCommon
from metrics.EMD import emd


class TrainerVAE(TrainerCommon):
	def __init__(self, config):
		super(TrainerVAE, self).__init__(config)
		self.reconstruction_loss = None
		self.kld_loss = None

	def build_model(self, config):
		# customize the build_model function to build the autoencoder
		model = get_model(config, "VAE")
		# print('#####-----pointAE architecture-----######')
		# print(model)
		model = model.to(self.device)
		return model

	def set_loss_function(self):
		self.criterion = emd()

	def forward(self, data, train=True):
		input_pts = data["points"].to(self.device)
		encoded_input_pts = data["points_encoded"].to(self.device)
		target_pts = input_pts.clone().transpose(1, 2)

		self.predicted_pts, mean, log_var = self.model(encoded_input_pts)
		if train:
			emd_dis, assignment = self.criterion(self.predicted_pts.transpose(1, 2), target_pts, 0.05, 3000)
			self.reconstruction_loss = torch.mean(torch.sqrt(emd_dis))
			self.kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
			self.loss = self.reconstruction_loss + self.kld_loss

	def collect_loss(self):
		loss_dict = {
			"comp_recon": self.reconstruction_loss,
			"kl_div": self.kld_loss
			}
		return loss_dict

	def visualize_batch(self, data, mode, num=2, **kwargs):
		tbw = self.train_tbw if mode == 'train' else self.val_tbw

		target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
		predicted_pts = self.predicted_pts[:num].transpose(1, 2).detach().cpu().numpy()

		tbw.add_mesh("gt", vertices=target_pts, global_step=self.watcher.step)
		tbw.add_mesh("predicted", vertices=predicted_pts, global_step=self.watcher.step)
