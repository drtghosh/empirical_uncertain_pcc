import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle, write_point_cloud_ply, create_grid, create_negative_with_label, write_ply

import os
from tqdm import tqdm

import numpy as np
import gpytorch
from gpytoolbox import write_mesh, fd_interpolate
from skimage.measure import marching_cubes

import seaborn as sns


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
                            "results_invest/ckpt-{}-n{}".format(config.ckpt, saved_test))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # test
    with torch.no_grad():
        for _ in tqdm(range(saved_test)):
            data = next(test_loader)

            trainer.use_complete_test = True
            trainer.forward(data, False)

            pc_dir = os.path.join(save_dir, trainer.data_id[0])
            if not os.path.exists(pc_dir):
                os.makedirs(pc_dir)
            # save the partial point cloud to results
            partial_points = trainer.partial_pc[0].transpose(1, 0).cpu().numpy()
            write_point_cloud_ply(partial_points, os.path.join(pc_dir, 'partial.ply'))
            # save the complete point cloud to results
            write_point_cloud_ply(trainer.complete_pc[0].transpose(1, 0).cpu().numpy(),
                                  os.path.join(pc_dir, 'complete.ply'))

            # create a grid around partial data
            grid_data, grid_sizes, corner, spacing = create_grid(trainer.partial_pc.transpose(1, 2), config.grid_size,
                                                trainer.partial_pc.size(1))
            grid_data = grid_data.to(trainer.device)
            # create a grid around complete data
            grid_data_c, grid_sizes_c, corner_c, spacing_c = create_grid(trainer.complete_pc.transpose(1, 2),
                                                                         config.grid_size, trainer.complete_pc.size(1))
            grid_data_c = grid_data_c.to(trainer.device)

            """
                Compare two grids and plot grid vertices along with the point clouds
            """
            grid_log_file = os.path.join(pc_dir, 'grid_compare.txt')
            with open(grid_log_file, 'w') as fg:
                fg.write(f"Corners of grid encompassing partial data: {corner, grid_data[0][-1]}!\n")
                fg.write("Grid range (partial):\n")
                fg.write(f"x axis: {spacing[0] * (grid_sizes[0] - 1)}\n")
                fg.write(f"y axis: {spacing[1] * (grid_sizes[1] - 1)}\n")
                fg.write(f"z axis: {spacing[2] * (grid_sizes[2] - 1)}\n")
                fg.write(f"Corners of grid encompassing complete data: {corner_c, grid_data_c[0][-1]}!\n")
                fg.write("Grid range (complete):\n")
                fg.write(f"x axis: {spacing_c[0] * (grid_sizes_c[0] - 1)}\n")
                fg.write(f"y axis: {spacing_c[1] * (grid_sizes_c[1] - 1)}\n")
                fg.write(f"z axis: {spacing_c[2] * (grid_sizes_c[2] - 1)}\n")

            # output embedding for grid points
            extended_grid = torch.cat([grid_data, trainer.test_latent.expand(-1, grid_data.size(1), -1)], 2)
            trainer.model.eval()
            grid_embedding = trainer.model(extended_grid).flatten(0, 1)

            # create negative data for the partial data
            negative_cloud, negative_label = create_negative_with_label(trainer.partial_pc.transpose(1, 2))
            negative_cloud = negative_cloud.to(trainer.device)
            # output embedding for negative data
            extended_negative = torch.cat([negative_cloud, trainer.test_latent.expand(-1, negative_cloud.size(1), -1)],
                                          2)
            trainer.model.eval()
            negative_embedding = trainer.model(extended_negative)

            """
                Compare embeddings of partial, complete, negative points
            """
            part_pair_distance = torch.cdist(trainer.partial_embedding, trainer.partial_embedding, p=2).flatten(0, 1)
            heatmap_part = sns.heatmap(part_pair_distance.cpu().numpy(), cbar=False)
            heatmap1 = heatmap_part.get_figure()
            heatmap1.savefig(os.path.join(pc_dir, 'heatmap_partial.jpg'))

            comp_pair_distance = torch.cdist(trainer.complete_embedding, trainer.complete_embedding, p=2).flatten(0, 1)
            heatmap_comp = sns.heatmap(comp_pair_distance.cpu().numpy(), cbar=False)
            heatmap2 = heatmap_comp.get_figure()
            heatmap2.savefig(os.path.join(pc_dir, 'heatmap_complete.jpg'))

            neg_pair_distance = torch.cdist(negative_embedding, negative_embedding, p=2).flatten(0, 1)
            heatmap_neg = sns.heatmap(neg_pair_distance.cpu().numpy(), cbar=False)
            heatmap3 = heatmap_neg.get_figure()
            heatmap3.savefig(os.path.join(pc_dir, 'heatmap_negative.jpg'))

            part_comp_distance = torch.cdist(trainer.partial_embedding, trainer.complete_embedding, p=2).flatten(0, 1)
            heatmap_pc = sns.heatmap(part_comp_distance.cpu().numpy(), cbar=False)
            heatmap4 = heatmap_pc.get_figure()
            heatmap4.savefig(os.path.join(pc_dir, 'heatmap_part_vs_comp.jpg'))

            part_neg_distance = torch.cdist(trainer.partial_embedding, negative_embedding, p=2).flatten(0, 1)
            heatmap_pn = sns.heatmap(part_neg_distance.cpu().numpy(), cbar=False)
            heatmap5 = heatmap_pn.get_figure()
            heatmap5.savefig(os.path.join(pc_dir, 'heatmap_part_vs_neg.jpg'))

            comp_neg_distance = torch.cdist(trainer.complete_embedding, negative_embedding, p=2).flatten(0, 1)
            heatmap_cn = sns.heatmap(comp_neg_distance.cpu().numpy(), cbar=False)
            heatmap6 = heatmap_cn.get_figure()
            heatmap6.savefig(os.path.join(pc_dir, 'heatmap_comp_vs_neg.jpg'))

            # combine test embeddings
            test_embedding = torch.cat([trainer.partial_embedding, negative_embedding], 1).flatten(0, 1)

            # combine test labels
            test_label = torch.concat((torch.zeros(trainer.partial_pc.size(-1)), negative_label), 0).to(trainer.device)

            # gaussian process
            cov_fn = gpytorch.kernels.RBFKernel(ard_num_dims=test_embedding.size(-1)).to(trainer.device)
            cov_pp = cov_fn(test_embedding).evaluate_kernel().to_dense()
            heatmap_pp = sns.heatmap(cov_pp.cpu().numpy(), cbar=False)
            heatmap7 = heatmap_pp.get_figure()
            heatmap7.savefig(os.path.join(pc_dir, 'heatmap_pp.jpg'))
            additional_noise = config.noise_variance * torch.eye(test_embedding.size(0)).to(trainer.device)
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
                # if (i+1) % 32 == 0:
                '''if 1199 <= i <= 1263:
                    heatmap_pb = sns.heatmap(cov_pb.cpu().numpy(), cbar=False)
                    heatmap8 = heatmap_pb.get_figure()
                    heatmap8.savefig(os.path.join(pc_dir, f'heatmap_pb{i}.jpg'))'''
                cov_bb = cov_fn(b, b).evaluate_kernel().to_dense()
                posterior_mean = 1 + cov_pb.T @ cov_inv @ (test_label - 1)
                posterior_var = cov_bb - cov_pb.T @ cov_inv @ cov_pb
                posterior_diag = torch.diagonal(posterior_var, 0)
                grid_posterior_mean[i * config.gp_batch: (i + 1) * config.gp_batch] = posterior_mean
                grid_posterior_var[i * config.gp_batch: (i + 1) * config.gp_batch] = posterior_diag

            # shift posterior mean
            W = fd_interpolate(partial_points, grid_sizes, spacing, corner)
            shift = np.sum(W @ grid_posterior_mean.cpu().numpy()) / partial_points.shape[0]
            shifted_mean = grid_posterior_mean.cpu().numpy() - shift
            # marching cubes
            vertices, faces, normals, values = marching_cubes(np.reshape(shifted_mean, grid_sizes, order='F'),
                                                              level=0.0)
            W2 = fd_interpolate(vertices, grid_sizes, spacing, corner)
            var_on_vertices = W2 @ grid_posterior_var.cpu().numpy()
            # save mesh into .obj file
            write_mesh(os.path.join(pc_dir, 'mean_shifted.obj'), vertices, faces)
            print(grid_posterior_var)
            write_ply(os.path.join(pc_dir, 'mean_shifted_with_color.ply'), vertices, faces, var_on_vertices)

            # without mean shifting
            vertices_og, faces_og, normals_og, values_og = marching_cubes(
                np.reshape(grid_posterior_mean.cpu().numpy(), grid_sizes, order='F'), level=0.0)
            write_mesh(os.path.join(pc_dir, 'mean_og.obj'), vertices_og, faces_og)

            # standard deviation plus (1 time)
            plus_std = shifted_mean + grid_posterior_var.cpu().numpy()
            vertices_p, faces_p, normals_p, values_p = marching_cubes(np.reshape(plus_std, grid_sizes, order='F'),
                                                              level=0.0)
            write_mesh(os.path.join(pc_dir, 'std_plus.obj'), vertices_p, faces_p)

            # standard deviation minus (1 time)
            minus_std = shifted_mean - grid_posterior_var.cpu().numpy()
            vertices_m, faces_m, normals_m, values_m = marching_cubes(np.reshape(minus_std, grid_sizes, order='F'),
                                                                      level=0.0)
            write_mesh(os.path.join(pc_dir, 'std_minus.obj'), vertices_m, faces_m)


if __name__ == '__main__':
    test_con()
