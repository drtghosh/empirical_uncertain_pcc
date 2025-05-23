import torch
from models import get_model
from tools.base_trainer import TrainerCommonINR
from metrics import hessMorse


class TrainerHessian(TrainerCommonINR):
    def build_model(self, config):
        # customize the build_model function to build the autoencoder
        if config.simple_hessian:
            model = get_model(config, "HessSimple")
        else:
            model = get_model(config, "HessComplex")
        model = model.to(self.device)
        return model

    def set_loss_function(self, config):
        self.criterion = hessMorse(weights=config.loss_weights, div_decay=config.morse_decay,
                                   div_type=config.morse_type, bidirectional_morse=config.bidirectional_morse)

    def forward(self, data, train=True):
        partial_pc, partial_enc, complete_pc, non_manifold_pts, near_pts = (
            data['partial_points'].to(self.device), data['partial_encoded'].to(self.device),
            data['gt_points'].to(self.device), data['non_manifold_points'].to(self.device),
            data['near_points'].to(self.device))

        if train:
            manifold_pts = complete_pc

            manifold_pts.requires_grad_()
            non_manifold_pts.requires_grad_()
            near_pts.requires_grad_()

            output_pred = self.model(partial_enc, manifold_pts, non_manifold_pts, near_pts)

            self.loss, self.loss_dict, _ = self.criterion(output_pred, manifold_pts, non_manifold_pts, near_pts)
        else:
            manifold_pts = partial_pc

    def collect_loss(self):
        return self.loss_dict
