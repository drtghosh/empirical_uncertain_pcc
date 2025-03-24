import torch
from models import get_model, set_requires_grad
from tools.base_trainer import TrainerContrastive
from metrics import Triplet, lacc


class TrainerAEContrast(TrainerContrastive):
    def __init__(self, config):
        super(TrainerAEContrast, self).__init__(config)
        self.partial_pc = None
        self.complete_pc = None
        self.data_id = None
        self.partial_embedding = None
        self.test_latent = None

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

    def forward(self, data, train=True):
        self.data_id = data['id']
        self.partial_pc = data['partial_points'].to(self.device)
        partial_enc = data['partial_encoded'].to(self.device)
        self.complete_pc = data['gt_points'].to(self.device)

        with torch.no_grad():
            partial_latent = self.pointAE.encode(partial_enc)
            partial_latent = partial_latent.view(partial_latent.size(0),partial_latent.size(1), 1)

        extended_anchor = torch.cat([self.partial_pc, partial_latent.expand(-1, -1, self.partial_pc.size(-1))], 1)
        extended_anchor = extended_anchor.transpose(1, 2)
        if train:
            negative_pc = data['negative_points'].to(self.device)
            anchor_embedding = self.model(extended_anchor).flatten(0, 1)
            idx_pair = torch.randperm(self.complete_pc.size(-1))
            subsample_idx_pair = idx_pair[:extended_anchor.size(1)]
            sub_complete = self.complete_pc[:, :, subsample_idx_pair]
            sub_negative = negative_pc[:, :, subsample_idx_pair]

            extended_complete = torch.cat([sub_complete, partial_latent.expand(-1, -1, sub_complete.size(-1))], 1)
            extended_complete = extended_complete.transpose(1, 2)
            extended_negative = torch.cat([sub_negative, partial_latent.expand(-1, -1, sub_negative.size(-1))], 1)
            extended_negative = extended_negative.transpose(1, 2)

            positive_embedding = self.model(extended_complete).flatten(0, 1)
            negative_embedding = self.model(extended_negative).flatten(0, 1)
            self.loss = lacc(anchor_embedding, positive_embedding, negative_embedding, self.loss_batch,
                         self.criterionContrast)
        else:
            self.model.eval()
            with torch.no_grad():
                self.partial_embedding = self.model(extended_anchor)
                self.test_latent = partial_latent

    def collect_loss(self):
        loss_dict = {
            "contrastive_loss": self.loss
        }
        return loss_dict
