import torch
from models import get_model, set_requires_grad
from tools.base_trainer import TrainerCommonMulti
from metrics import ldf
from models import enable_dropout


class TrainerMCDropout(TrainerCommonMulti):
    def __init__(self, config):
        super(TrainerMCDropout, self).__init__(config)
        self.partial_pc = None
        self.complete_pc = None
        self.data_id = None
        self.latent_gen_loss = None
        self.recon_fidelity_loss = None
        self.recon_weight = config.recon_weight
        self.latent_gen_weight = config.latent_gen_weight
        if not config.is_train:
            self.n_samples = config.gen_samples_test
            self.latent_gen_list = []
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
        model = get_model(config, "genDropout").to(self.device)
        return model

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

        if train:
            latent_gen = self.model(partial_latent)
            self.gen_pc = self.pointAE.decode(latent_gen)

            # compute loss
            self.latent_gen_loss = self.latent_gen_weight * self.criterionLatent(latent_gen, complete_latent)
            self.recon_fidelity_loss = self.recon_weight * ldf(self.partial_pc, self.gen_pc)
            self.loss = self.latent_gen_loss + self.recon_fidelity_loss
        else:
            self.latent_gen_list = []
            for idx in range(self.n_samples):
                self.model.eval()
                enable_dropout(self.model)
                with torch.no_grad():
                    latent_gen = self.model(partial_latent)
                self.latent_gen_list.append(latent_gen)

    def collect_loss(self):
        loss_dict = {
            "latent_mse": self.latent_gen_loss,
            "part_recon": self.recon_fidelity_loss
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
