from models.autoencoder import PointAE
from models.vae import VAE
from models.vqvae import VQVAE
from models.multimodal_imle import Generator
from models.dropout_generator import GeneratorDropout
from models.dropconnect_generator import GeneratorDropConnect
from models.multimodal_ebm import EBMCompletion
from models.generic_models import MLPBlock, MLPConv
from models.model_utils import gen_nearest_latents


def get_model(config, name):
    if name == "pointAE":
        return PointAE(config)
    elif name == "Gen":
        return Generator(config)
    elif name == "genDropout":
        return GeneratorDropout(config)
    elif name == "genDropCon":
        return GeneratorDropConnect(config)
    elif name == "EBMgen":
        return EBMCompletion(config)
    elif name == "blockMLP":
        return MLPBlock(config)
    elif name == "VAE":
        return VAE(config)
    elif name == "VQVAE":
        return VQVAE(config)
    else:
        raise NotImplementedError("Got name '{}'".format(name))


def set_requires_grad(models, requires_grad=False):
    """
        Parameters:
            models (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if models is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad
