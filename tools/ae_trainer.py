import torch
import torch.nn as nn
from models import get_model
from tools.base_trainer import TrainerCommon


class TrainerAE(TrainerCommon):
    def build_model(self, config):
        # customize the build_model function
        model = get_model(config, "pointAE")
        # print('#####-----pointAE architecture-----######')
        # print(model)
        model = model.cuda()
        return model
