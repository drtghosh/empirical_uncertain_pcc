import torch
from models import get_model
from tools.base_trainer import TrainerCommon
from metrics.EMD import emd


class TrainerHessian(TrainerCommon):
    def build_model(self, config):
        # customize the build_model function to build the autoencoder
        model = get_model(config, "Hess")
        model = model.to(self.device)
        return model

    def set_loss_function(self):
        self.criterion = emd()

    def forward(self, data, train=True):
        partial_pc, partial_enc, complete_pc, complete_enc, non_manifold_pts, near_pts = (
            data['partial_points'].to(self.device), data['partial_encoded'].to(self.device),
            data['gt_points'].to(self.device), data['gt_encoded'].to(self.device),
            data['non_manifold_points'].to(self.device), data['near_points'].to(self.device))

        manifold_pts = partial_pc
        if train:
            manifold_pts = complete_pc

        manifold_pts.requires_grad()
        non_manifold_pts.requires_grad()
        near_pts.requires_grad()


    def collect_loss(self):
        loss_dict = {"emd": self.loss}
        return loss_dict
