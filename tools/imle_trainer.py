import torch
from models import get_model, set_requires_grad
from tools.base_trainer import TrainerCommon
from torch.distributions import normal
from dciknn_cuda import DCI


class TrainerIMLE(TrainerCommon):
    def __init__(self, config):
        super(TrainerIMLE, self).__init__(config)
        self.pointAE = None
        self.generator = None
        self.part_recon_loss = None
        self.z_dim = config.noise_dim
        self.z_samples_train = config.gen_samples_train
        self.z_samples_test = config.gen_samples_test
        self.z_sampler = normal.Normal(0, 1)
        self.dci_db = DCI(128, 2, 10, 100, 10)

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
        self.generator = get_model(config, "Gen").to(self.device)

    def collect_loss(self):
        loss_dict = {
            "imle": self.criterion,
            "part_recon": self.part_recon_loss}
        return loss_dict

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
        self.fake_latent = gen_samples(self.dci_db, latent_gen_list, complete_latent)
        self.fake_pc = self.pointAE.decode(self.fake_latent)
