"""import torch
from models import get_model, set_requires_grad
from tools.base_trainer import TrainerContrastive
from metrics import Triplet


class TrainerAEContrast(TrainerContrastive):
    def __init__(self, config):
        super(TrainerAEContrast, self).__init__(config)
        self.pointAE = None

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

        # customize the build_model function to build contrastive encoder
        contrastive_encoder = get_model(config, "blockMLP").to(self.device)
        return contrastive_encoder

    def set_loss_function(self):
        self.criterionContrast = Triplet()

    def forward(self, data):
        partial_pc = data['partial_points'].to(self.device)
        complete_pc = data['gt_points'].to(self.device)

        with torch.no_grad():
            partial_latent = self.pointAE.encode(partial_pc)

        extended_anchor = torch.cat([partial_pc, partial_latent.expand(-1, partial_pc.size(1), -1)], 2)
        idx_pair = torch.randperm(complete_pc.size(1))
        positive = subsample_idx_pair = idx_pair[:extended_anchor.size(-1)]

        conditioned_complete = torch.cat([complete_pc, partial_latent.expand(-1, complete_pc.size(1), -1)], 2)
"""
