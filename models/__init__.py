from models.autoencoder import PointAE


def get_model(config, name):
    if name == "pointAE":
        return PointAE(config)
    else:
        raise NotImplementedError("Got name '{}'".format(name))
