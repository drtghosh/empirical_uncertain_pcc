from tools.ae_trainer import TrainerAE


def get_trainer(config):
    if config.module == 'ae':
        return TrainerAE(config)
    else:
        raise ValueError
