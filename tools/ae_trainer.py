import torch
from models import get_model
from tools.base_trainer import TrainerCommon
from metrics.EMD import emd


class TrainerAE(TrainerCommon):
    def build_model(self, config):
        # customize the build_model function to build the autoencoder
        model = get_model(config, "pointAE")
        # print('#####-----pointAE architecture-----######')
        # print(model)
        model = model.to(self.device)
        return model

    def set_loss_function(self):
        self.criterion = emd()

    def forward(self, data, train=True):
        input_pts = data["points"].to(self.device)
        encoded_input_pts = data["points_encoded"].to(self.device)
        target_pts = input_pts.clone().detach()

        self.predicted_pts = self.model(encoded_input_pts)
        if train:
            self.loss = torch.mean(self.criterion(self.predicted_pts, target_pts))

    def collect_loss(self):
        loss_dict = {"emd": self.loss}
        return loss_dict

    def visualize_batch(self, data, mode, num=2, **kwargs):
        tbw = self.train_tbw if mode == 'train' else self.val_tbw

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        predicted_pts = self.predicted_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tbw.add_mesh("gt", vertices=target_pts, global_step=self.watcher.step)
        tbw.add_mesh("predicted", vertices=predicted_pts, global_step=self.watcher.step)
