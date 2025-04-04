from tools.ae_trainer import TrainerAE
from tools.vae_trainer import TrainerVAE
from tools.vqvae_trainer import TrainerVQVAE
from tools.imle_trainer import TrainerIMLE
from tools.contrastive_trainer import TrainerAEContrast, TrainerVQVAEContrast


def get_trainer(config):
    if config.module == 'ae':
        return TrainerAE(config)
    elif config.module == 'vae':
        return TrainerVAE(config)
    elif config.module == 'vqvae':
        return TrainerVQVAE(config)
    elif config.module == 'imle_gen':
        return TrainerIMLE(config)
    elif config.module == 'contrast_ae' or config.module == 'contrast_all_ae':
        return TrainerAEContrast(config)
    elif config.module == 'contrast_vqvae':
        return TrainerVQVAEContrast(config)
    else:
        raise ValueError
