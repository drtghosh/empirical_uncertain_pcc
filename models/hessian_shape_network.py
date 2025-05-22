import torch.nn as nn

from .filmsiren import FilmSiren
from .convolutional_encoder import ConvolutionalFeature


class ShapeNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ConvolutionalFeature()
        # decoder_hidden_dim=256, decoder_n_hidden_layers=5
        self.decoder = FilmSiren(hidden_size=config.decoder_hidden_dim, n_layers=config.decoder_n_hidden_layers)

    def forward(self, non_manifold_pts=None, manifold_pts=None, near_pts=None):
        return self.forward_dense(non_manifold_pts, manifold_pts, near_pts)

    def forward_dense(self, non_manifold_pts=None, manifold_pts=None, near_pts=None):
        latent_reg = None
        # manifold
        global_feat = self.encoder.encode(manifold_pts)
        manifold_pts_feat = self.encoder.query_feature(global_feat, manifold_pts)
        manifold_pts_pred = self.decoder(manifold_pts, manifold_pts_feat)
        # non-manifold
        non_manifold_pts_pred = None
        if non_manifold_pts is not None:
            non_manifold_nts_feat = self.encoder.query_feature(global_feat, non_manifold_pts)
            non_manifold_pts_pred = self.decoder(non_manifold_pts, non_manifold_nts_feat)

        near_pts_pred = None
        if near_pts is not None:
            near_pts_feat = self.encoder.query_feature(global_feat, near_pts)
            near_pts_pred = self.decoder(near_pts, near_pts_feat)

        return {"manifold_pts_pred": manifold_pts_pred,
                "non_manifold_pts_pred": non_manifold_pts_pred,
                'near_pts_pred': near_pts_pred,
                "latent_reg": latent_reg,
                }
