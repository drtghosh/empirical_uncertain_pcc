import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle, write_point_cloud_ply, create_grid, create_negative_with_label

import os
from tqdm import tqdm

import numpy as np
import gpytorch
from gpytoolbox import write_mesh
from skimage.measure import marching_cubes


def test_con():
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
                            "results/ckpt-{}-n{}".format(config.ckpt, saved_test))
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
        # save the partial point cloud to results
        write_point_cloud_ply(trainer.partial_pc[0].transpose(1, 0).cpu().numpy(), os.path.join(pc_dir, 'partial.ply'))
        # save the complete point cloud to results
        write_point_cloud_ply(trainer.complete_pc[0].transpose(1, 0).cpu().numpy(),
                              os.path.join(pc_dir, 'complete.ply'))

        # create a grid around partial data
        grid_data, grid_sizes = create_grid(trainer.partial_pc.transpose(1, 2), config.grid_size,
                                            trainer.partial_pc.size(1))
        grid_data.to(trainer.device)
        # output embedding for grid points
        extended_grid = torch.cat([grid_data, trainer.test_latent.expand(-1, grid_data.size(1), -1)], 2)
        trainer.model.eval()
        with torch.no_grad():
            grid_embedding = trainer.model(extended_grid)

        # create negative data for the partial data
        negative_cloud, negative_label = create_negative_with_label(trainer.partial_pc.transpose(1, 2))
        negative_cloud.to(trainer.device)
        negative_label.to(trainer.device)
        # output embedding for negative data
        extended_negative = torch.cat([negative_cloud, trainer.test_latent.expand(-1, negative_cloud.size(1), -1)], 2)
        trainer.model.eval()
        with torch.no_grad():
            negative_embedding = trainer.model(extended_negative)

        # combine test embeddings
        test_embedding = torch.cat([trainer.partial_embedding, negative_embedding], 1).flatten(0, 1)

        # combine test labels
        test_label = torch.concat((torch.zeros(trainer.partial_pc.size(-1)), negative_label), 0).to(trainer.device)

        # gaussian process
        cov_fn = gpytorch.kernels.RBFKernel(ard_num_dims=test_embedding.size(-1)).to(trainer.device)
        cov_pp = cov_fn(test_embedding).evaluate_kernel().to_dense()
        additional_noise = config.noise_variance * torch.eye(test_embedding.size(1)).to(trainer.device)
        cov_with_noise = (cov_pp + additional_noise)
        cov_inv = torch.linalg.inv(cov_with_noise)
        assert grid_embedding.size(
            0) % config.gp_batch == 0, 'Number of grid points required to be a multiple of batch size'
        num_batches = grid_embedding.size(0) // config.gp_batch
        grid_posterior_mean = torch.empty(grid_embedding.size(0))
        grid_posterior_var = torch.empty(grid_embedding.size(0))
        for i in range(num_batches):
            b = grid_embedding[i * config.gp_batch: (i + 1) * config.gp_batch]
            cov_pb = cov_fn(test_embedding, b).evaluate_kernel().to_dense()
            cov_bb = cov_fn(b, b).evaluate_kernel().to_dense()
            posterior_mean = cov_pb.T @ cov_inv @ test_label
            posterior_var = cov_bb - cov_pb.T @ cov_inv @ cov_pb
            posterior_diag = torch.diagonal(posterior_var, 0)
            grid_posterior_mean[i * config.gp_batch: (i + 1) * config.gp_batch] = posterior_mean
            grid_posterior_var[i * config.gp_batch: (i + 1) * config.gp_batch] = posterior_diag

        # marching cubes
        vertices, faces, normals, values = marching_cubes(
            np.reshape(grid_posterior_mean.cpu().detach().numpy(), grid_sizes, order='F'), level=0.0)
        # save mesh into .obj file
        write_mesh(os.path.join(pc_dir, 'mean.obj'), vertices, faces)


if __name__ == '__main__':
    test_con()
