import torch
from models import get_model
from tools.base_trainer import TrainerCommon


class TrainerIMLE(TrainerCommon):
    def __init__(self, config):
        super(TrainerIMLE, self).__init__(config)
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
        self.pointAE = self.pointAE.eval().cuda()
        set_requires_grad(self.pointAE, False)

        # build G, D
        self.netG = get_network(config, "G").cuda()
        self.l2_loss = nn.MSELoss()
