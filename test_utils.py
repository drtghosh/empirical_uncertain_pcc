import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

import cv2 as cv


def naive_estimation(generated_clouds):
    gen_mu = generated_clouds.mean(dim=0)
    gen_std = generated_clouds.std(dim=0)
    normalized_std = gen_std / gen_std.max()
    intensity = torch.norm(normalized_std, dim=[1])
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity = 255 - np.uint8(255 * intensity)
    color_map = cv.applyColorMap(intensity, cv.COLORMAP_JET) / 255
    color_map = np.squeeze(color_map)

    return gen_mu, color_map


def matching_estimation(generated_clouds, p=2):
    for i in range(1, len(generated_clouds)):
        cost_matrix = torch.cdist(generated_clouds[0], generated_clouds[i], p)
        _, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
        generated_clouds[i] = generated_clouds[i][col_ind, :]
    gen_mu = generated_clouds.mean(dim=0)
    gen_std = generated_clouds.std(dim=0)
    normalized_std = gen_std / gen_std.max()
    intensity = torch.norm(normalized_std, dim=[1])
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity = 255 - np.uint8(255 * intensity)
    color_map = cv.applyColorMap(intensity, cv.COLORMAP_JET) / 255
    color_map = np.squeeze(color_map)

    return gen_mu, color_map
