import torch
from models import get_model, set_requires_grad, gen_nearest_latents
from tools.base_trainer import TrainerCommonMulti
from torch.distributions import normal
from dciknn_cuda import DCI
from metrics import ldf


class TrainerIMLE(TrainerCommonMulti):
    def __init__(self, config):
        super(TrainerIMLE, self).__init__(config)
        self.pointAE = None
        self.latent_gen_loss = None
        self.reconstruction_loss = None
        self.recon_weight = config.recon_weight
        self.latent_gen_weight = config.latent_gen_weight
        self.z_dim = config.noise_dim
        self.z_samples_train = config.gen_samples_train
        self.z_samples_test = None
        if not config.is_train:
            self.z_samples_test = config.gen_samples_test
        self.z_sampler = normal.Normal(0, 1)
        # dci_db = DCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size, devices=[0, 1])
        self.dci_db = DCI(config.latent_dim, 2, 10, 100, 10)
        self.gen_pc = None

    def build_model(self, config):
        # load pretrained pointAE
        self.pointAE = get_model(config, "pointAE")
        # -------------------------config check tbd-------------------------
        try:
            ae_weights = torch.load(config.path_pretrained_ae)['model_state_dict']
        except Exception as e:
            raise ValueError(f"The path for pretrained autoencoder doesn't exist.\n{e}")
        self.pointAE.load_state_dict(ae_weights)
        self.pointAE = self.pointAE.eval().to(self.device)
        set_requires_grad(self.pointAE, False)

        # customize the build_model function to build generator
        generator = get_model(config, "Gen").to(self.device)
        return generator

    def set_loss_function(self):
        self.criterionLatent = self.criterionMSE

    def forward(self, data, train=True):
        partial_pc = data['partial_points'].to(self.device)
        complete_pc = data['gt_points'].to(self.device)

        with torch.no_grad():
            partial_latent = self.pointAE.encode(partial_pc)
            complete_latent = self.pointAE.encode(complete_pc)
        z_samples = self.z_samples_train if train else self.z_samples_test

        latent_gen_list = []

        for idx in range(z_samples):
            z_random = self.z_sampler.sample([partial_latent.size(0), self.z_dim]).to(self.device)
            latent_gen = self.generator(partial_latent, z_random)
            latent_gen_list.append(latent_gen)

        latent_gen_list = torch.stack(latent_gen_list, 1)

        # sample
        latent_gen_nearest = gen_nearest_latents(self.dci_db, latent_gen_list, complete_latent)
        self.gen_pc = self.pointAE.decode(latent_gen_nearest)

        # compute loss
        self.latent_gen_loss = self.latent_gen_weight * self.criterionLatent(latent_gen_nearest, complete_latent)
        self.reconstruction_loss = self.recon_weight * ldf(partial_pc, self.gen_pc)
        self.loss = self.latent_gen_loss + self.reconstruction_loss

    def collect_loss(self):
        loss_dict = {
            "imle": self.latent_gen_loss,
            "part_recon": self.reconstruction_loss
            }
        return loss_dict

    def update_generator(self):
        self.optimizer_gen.zero_grad()
        self.loss.backward()
        self.optimizer_gen.step()

    def visualize_batch(self, data, mode, num=2, **kwargs):
        tbw = self.train_tbw if mode == 'train' else self.val_tbw

        target_pts = data['gt_points'][:num].transpose(1, 2).detach().cpu().numpy()
        partial_pts = data['partial_points'][:num].transpose(1, 2).detach().cpu().numpy()
        generated_pts = self.gen_pc[:num].transpose(1, 2).detach().cpu().numpy()

        # generated_pts = np.clip(generated_pts, -0.999, 0.999)

        tbw.add_mesh("gt", vertices=target_pts, global_step=self.watcher.step)
        tbw.add_mesh("partial", vertices=partial_pts, global_step=self.watcher.step)
        tbw.add_mesh("generated", vertices=generated_pts, global_step=self.watcher.step)
