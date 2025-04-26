from tools.ae_trainer import TrainerAE
from tools.vae_trainer import TrainerVAE
from tools.vqvae_trainer import TrainerVQVAE
from tools.imle_trainer import TrainerIMLE
from tools.mcdropout_trainer import TrainerMCDropout
from tools.mcdropconnect_trainer import TrainerMCDropConnect
from tools.ensemble_trainer import TrainerDeepEnsemble
from tools.ebm_trainer import TrainerEBM
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
    elif config.module == 'dropout_gen':
        return TrainerMCDropout(config)
    elif config.module == 'drop_con_gen':
        return TrainerMCDropConnect(config)
    elif config.module == 'ensemble_gen':
        return TrainerDeepEnsemble(config)
    elif config.module == 'ebm_gen':
        return TrainerEBM(config)
    elif config.module == 'contrast_ae' or config.module == 'contrast_all_ae' or config.module == 'contrast_grid_ae':
        return TrainerAEContrast(config)
    elif config.module == 'contrast_vqvae':
        return TrainerVQVAEContrast(config)
    else:
        raise ValueError
