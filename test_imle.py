import torch

from configs import get_config
from tools import get_trainer
from datasets import get_dataloader
from datasets.data_utils import cycle

import os
import open3d as o3d
from tqdm import tqdm


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
        partial_pc = trainer.partial_pc[0].transpose(1, 0)
        complete_pc = trainer.complete_pc[0].transpose(1, 0)
        pc_dir = os.path.join(save_dir, trainer.data_id)
        if not os.path.exists(pc_dir):
            os.makedirs(pc_dir)
        # save the partial point cloud to results
        o3d.io.write_point_cloud(os.path.join(pc_dir, 'partial.ply'), partial_pc)
        # save the complete point cloud to results
        o3d.io.write_point_cloud(os.path.join(pc_dir, 'complete.ply'), complete_pc)
        # save the generated point clouds to results
        for j in range(len(trainer.latent_gen_list)):
            latent = trainer.latent_gen_list[j]
            gen_pc = trainer.pointAE.decode(latent)[0].transpose(1, 0)
            o3d.io.write_point_cloud(os.path.join(pc_dir, f'gen_{j}.ply'), gen_pc)


if __name__ == '__main__':
    test_imle_gen()
