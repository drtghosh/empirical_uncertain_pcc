import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle, write_point_cloud_ply, plot_pcd_one_view

import os
from tqdm import tqdm

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


def test_imle_gen():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create model and trainer
    trainer = get_trainer(config)

    # load from checkpoint
    trainer.load_ckpt(config.ckpt)
    trainer.model.eval()

    # create dataloader
    config.batch_size = 1
    config.num_workers = 1
    test_loader = get_dataloader('test', config)
    num_test = len(test_loader)
    print(f"Total number of test samples: {num_test}.")
    saved_test = num_test if config.num_sample == -1 else config.num_sample
    print(f"Selected number of test samples to be saved: {saved_test}.")
    test_loader = cycle(test_loader)

    # directory to save results
    save_dir = os.path.join(config.proj_dir,
                            "results/ckpt-{}-n{}-z{}".format(config.ckpt, saved_test, config.gen_samples_test))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # test
    for _ in tqdm(range(saved_test)):
        data = next(test_loader)
        with torch.no_grad():
            trainer.forward(data, False)

        pc_dir = os.path.join(save_dir, trainer.data_id[0])
        if not os.path.exists(pc_dir):
            os.makedirs(pc_dir)

        # store all point clouds
        point_cloud_list = []
        titles = []
        colors = []

        # save the partial point cloud to results
        write_point_cloud_ply(trainer.partial_pc[0].transpose(1, 0).cpu().numpy(), os.path.join(pc_dir, 'partial.ply'))
        point_cloud_list.append(trainer.partial_pc[0].transpose(1, 0).cpu().numpy())
        titles.append('Partial Cloud')
        colors.append('red')
        # save the complete point cloud to results
        write_point_cloud_ply(trainer.complete_pc[0].transpose(1, 0).cpu().numpy(),
                              os.path.join(pc_dir, 'complete.ply'))
        point_cloud_list.append(trainer.complete_pc[0].transpose(1, 0).cpu().numpy())
        titles.append('Complete Cloud')
        colors.append('green')
        # save the generated point clouds to results
        # store to estimate confidence
        num_gen = len(trainer.latent_gen_list)
        gen_clouds = torch.zeros(num_gen, trainer.complete_pc.size(-1), trainer.complete_pc.size(1))
        for j in range(num_gen):
            latent = trainer.latent_gen_list[j]
            gen_pc_tensor = trainer.pointAE.decode(latent)[0].transpose(1, 0)
            gen_clouds[j] = gen_pc_tensor
            gen_pc = gen_pc_tensor.cpu().numpy()
            write_point_cloud_ply(gen_pc, os.path.join(pc_dir, f'gen_{j}.ply'))
            point_cloud_list.append(gen_pc)
            titles.append(f'Generated Cloud {j}')
            colors.append('blue')

        # estimate confidence
        # naive estimation
        gen_mu_naive, color_map_naive = naive_estimation(gen_clouds)
        write_point_cloud_ply(gen_mu_naive, os.path.join(pc_dir, 'gen_conf_naive.ply'), True, color_map_naive)
        point_cloud_list.append(gen_mu_naive.cpu().numpy())
        titles.append('Index-wise Est. Cloud')
        colors.append(color_map_naive)
        # linear assignment estimation
        gen_mu_matched, color_map_matched = matching_estimation(gen_clouds)
        write_point_cloud_ply(gen_mu_matched, os.path.join(pc_dir, 'gen_conf_lin.ply'), True, color_map_matched)
        point_cloud_list.append(gen_mu_matched.cpu().numpy())
        titles.append('Lin. Assign. Est. Cloud')
        colors.append(color_map_matched)

        plot_pcd_one_view(os.path.join(pc_dir, 'all.jpg'), point_cloud_list, titles, colors=colors)


if __name__ == '__main__':
    test_imle_gen()
