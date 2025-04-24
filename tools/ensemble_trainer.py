import torch
import torch.optim as optim
from models import get_model, set_requires_grad
from tools.base_trainer import TrainerCommonEnsemble
from metrics import ldf


class TrainerDeepEnsemble(TrainerCommonEnsemble):
    def __init__(self, config):
        super(TrainerDeepEnsemble, self).__init__(config)
        self.partial_pc = None
        self.complete_pc = None
        self.data_id = None
        self.latent_gen_losses = []
        self.recon_fidelity_losses = []
        self.recon_weight = config.recon_weight
        self.latent_gen_weight = config.latent_gen_weight
        if not config.is_train:
            self.latent_gen_list = []
        self.gen_pc = None

    def build_models(self, config):
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

        # customize the build_model function to build multiple generators for bagging
        models = []
        for i in range(self.n_models):
            model = get_model(config, "genEns").to(self.device)
            models.append(model)
        return models

    def set_optimizers(self, config):
        self.base_lr = config.lr
        optimizers = []
        for model in self.models:
            optimizer_gen = optim.Adam(model.parameters(), config.lr, betas=(config.beta1_gen, 0.999))
            optimizers.append(optimizer_gen)

    def set_loss_function(self):
        self.criterionLatent = self.criterionMSE

    def forward(self, data, train=True):
        self.data_id = data['id']
        self.partial_pc = data['partial_points'].to(self.device)
        partial_enc = data['partial_encoded'].to(self.device)
        self.complete_pc = data['gt_points'].to(self.device)
        complete_enc = data['gt_encoded'].to(self.device)

        with torch.no_grad():
            partial_latent = self.pointAE.encode(partial_enc)
            complete_latent = self.pointAE.encode(complete_enc)

        self.latent_gen_losses = []
        self.recon_fidelity_losses = []
        self.losses = []
        if train:
            for model in self.models:
                latent_gen = model(partial_latent)
                self.gen_pc = self.pointAE.decode(latent_gen)

                # compute loss
                latent_gen_loss = self.latent_gen_weight * self.criterionLatent(latent_gen, complete_latent)
                recon_fidelity_loss = self.recon_weight * ldf(self.partial_pc, self.gen_pc)
                combined_loss = latent_gen_loss + recon_fidelity_loss
                self.latent_gen_losses.append(latent_gen_loss)
                self.recon_fidelity_losses.append(recon_fidelity_loss)
                self.losses.append(combined_loss)
        else:
            self.latent_gen_list = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    latent_gen = model(partial_latent, train=True)
                self.latent_gen_list.append(latent_gen)

    def collect_losses(self):
        loss_dict = {
            "latent_mse": self.latent_gen_losses,
            "part_recon": self.recon_fidelity_losses
            }
        return loss_dict

    def update_generators(self):
        for i in range(self.n_models):
            self.optimizers_gen[i].zero_grad()
            self.losses[i].backward()
            self.optimizers_gen[i].step()
