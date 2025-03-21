from tools.ae_trainer import TrainerAE
from tools.imle_trainer import TrainerIMLE
from tools.contrastive_trainer import TrainerAEContrast


def get_trainer(config):
    if config.module == 'ae':
        return TrainerAE(config)
    elif config.module == 'imle_gen':
        return TrainerIMLE(config)
    elif config.module == 'contrast_ae':
        return TrainerAEContrast(config)
    else:
        raise ValueError
