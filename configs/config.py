import os
import argparse
import json
import shutil


def make_dir(dir_path):
    """
       create path by first checking its existence,
       :param dir_path: path
       :return:
       """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_dirs(dir_paths):
    """
    create paths by first checking their existence
    :param dir_paths: list of paths
    :return:
    """
    if isinstance(dir_paths, list) and not isinstance(dir_paths, str):
        for path in dir_paths:
            make_dir(path)
    else:
        make_dir(dir_paths)


def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """
        Base class of Config, provides necessary hyperparameters.
    """

    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("#########-----Experiment Configuration-----########")
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        # creating experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name, self.module)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite to retrain? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.result_dir = os.path.join(self.exp_dir, 'results')
        make_dirs([self.log_dir, self.model_dir, self.result_dir])

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """
        initializes argument parser. Define default hyperparameters and collect from command-line arguments.
        """
        parser = argparse.ArgumentParser()

        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_model_config_(parser)

        # training configuration
        self._add_training_config_(parser)

        if not self.is_train:
            # testing configuration
            self._add_testing_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    @staticmethod
    def _add_basic_config_(parser):
        """
        adds general arguments/ hyperparameters
        """
        group = parser.add_argument_group('basic')
        group.add_argument('-x', '--exp_name', type=str, help="Tag of experiment", required=True)
        group.add_argument('-l', '--proj_dir', type=str, default="proj_logger",
                           help="Path to directory where experiment logs/models will be saved")
        group.add_argument('-d', '--device', type=str, default='cuda:0', help='Device for training/ inference')
        group.add_argument('-m', '--module', type=str,
                           choices=['ae', 'vae', 'vqvae', 'c_gan', 'imle_gen', 'dropout_gen', 'drop_con_gen',
                                    'ensemble_gen', 'ebm_gen', 'contrast_ae', 'contrast_vqvae', 'contrast_all_ae',
                                    'contrast_grid_ae', 'hessian'],
                           required=True, help="Choice of the model to be used")

    @staticmethod
    def _add_dataset_config_(parser):
        """
        adds hyperparameters for dataset configuration
        """
        group = parser.add_argument_group('dataset')
        group.add_argument('-n', '--dataset_name', type=str, choices=['buildingpcc', 'pcn'], required=True,
                           help="Dataset to be used")
        group.add_argument('-r', '--data_root', type=str, default="", help="Path to corresponding data")
        group.add_argument('-f', '--data_file', type=str, default="", help="Name of file containing data split info")
        group.add_argument('-b', '--batch_size', type=int, default=8, help="Batch size for data loader")
        group.add_argument('-c', '--category', type=str, default="all", help="Shape category name")
        group.add_argument('-w', '--num_workers', type=int, default=8, help="Number of workers for data loader")
        group.add_argument('-p', '--n_pts', type=int, default=2048, help="Number of points sampled for complete shape. "
                                                                         "Half for partial if not specified otherwise")
        group.add_argument('--partial_pts', type=int, help="Number of points sampled for partial shape")

    def _add_model_config_(self, parser):
        """
        adds hyperparameters for model architecture
        """
        group = parser.add_argument_group('model')
        # ae encoder
        self.enc_features = (64, 128, 128, 256)
        self.res_layers = (2,)
        group.add_argument('--latent_dim', type=int, default=128)
        group.add_argument('--enc_norm', type=bool, default=True)
        group.add_argument('--space_dim', type=int, default=3)

        # ae decoder
        self.dec_features = (256, 256)
        group.add_argument('--dec_norm', type=bool, default=False)
        group.add_argument('--out_pts', type=int, default=2048, help="Number of points generated by decoder")

        # vqvae quantizer
        group.add_argument("--n_latent", type=int, default=32)
        group.add_argument("--beta_commit", type=float, default=0.25)

        # generator
        self.n_features_gen = (256, 512)
        group.add_argument('-a', '--path_pretrained_ae', type=str)
        group.add_argument('--noise_dim', type=int, default=32)
        group.add_argument('--gen_norm', type=bool, default=False)
        group.add_argument('--latent_gen_weight', type=float, default=5.0)
        group.add_argument('--recon_weight', type=float, default=6.0)
        group.add_argument('--gen_samples_train', type=int, default=20)
        
        # generator with dropout / dropconnect
        group.add_argument('--dropout_prob', type=float, default=0.1)

        # ensemble of generators
        group.add_argument('--n_models', type=int, default=10)

        # energy based model
        group.add_argument('--step_size', type=float, default=0.05)
        group.add_argument('--n_step', type=int, default=8)
        group.add_argument('--noise_scale', type=float, default=0.0001)
        group.add_argument('--recon_weight_ebm', type=float, default=1.0)
        group.add_argument('--fidelity_weight_ebm', type=float, default=2.0)
        group.add_argument('--latent_gen_weight_ebm', type=float, default=1.0)
        group.add_argument('--regularization_weight_ebm', type=float, default=0.1)

        # contrastive learning
        self.mlp_nodes = 32
        self.mlp_layers = 2
        group.add_argument('--contrast_dim', type=int, default=16)
        group.add_argument('--loss_contrastive', type=str, choices=['triplet', 'spacetime'], default='triplet')
        group.add_argument('--triplet_margin', type=float, default=0.1)
        group.add_argument('--loss_batch', type=int, default=512)

        # hessian inr
        parser.add_argument('--seed', type=int, default=3627473, help='random seed')
        parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
        self.enc_features_inr = (64, 128, 128, 256)
        self.res_layers_inr = (2,)
        self.dec_features_inr = (512, 512, 512)
        self.loss_weights = [7e3, 6e2, 5e1, 3, 1]
        parser.add_argument('--simple_hessian', type=bool, default=True, help='whether to use simple network')
        parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
        parser.add_argument('--morse_decay', type=str, default='linear',
                            help='divergence term importance decay none | step | linear')
        parser.add_argument('--bidirectional_morse', action='store_true', default=True)
        parser.add_argument('--decay_params', nargs='+', type=float, default=[3, 0.1, 3, 0.2, 0.001, 0],
                            help='epoch number to evaluate')

    @staticmethod
    def _add_training_config_(parser):
        """
        training configuration
        """
        group = parser.add_argument_group('training')
        group.add_argument('-e', '--num_epochs', type=int, default=1000, help="Epochs of training")
        group.add_argument('--lr', type=float, default=5e-4, help="Initial learning rate")
        group.add_argument('--lr_decay', type=float, default=0.9995, help="Step size for learning rate decay")
        group.add_argument('--decay_step', type=int, default=21, help="Decay steps for the LambdaLR scheduler")
        group.add_argument('--lowest_decay', type=float, default=0.02, help="Lowest decay for the LambdaLR scheduler")
        group.add_argument('--beta1_gen', type=float, default=0.5, help="beta1 for Adam when training generator")
        group.add_argument('--beta1_ed', type=float, default=0.9, help="beta1 for Adam when training ebm En/Decoder")
        group.add_argument('--beta1_ebm', type=float, default=0.9, help="beta1 for Adam when training ebm module")
        group.add_argument('--beta1_con', type=float, default=0.9, help="beta1 for Adam in contrastive learning")
        group.add_argument('--continue', dest='cont', action='store_true', help="Continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="Desired checkpoint to restore")
        group.add_argument('--vis', action='store_true', default=False, help="Visualize output in tensorboard")
        group.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
        group.add_argument('--save_frequency', type=int, default=100, help="Save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=10, help="Run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=100, help="Visualize output every x iterations")

    @staticmethod
    def _add_testing_config_(parser):
        """
            testing configuration
        """
        group = parser.add_argument_group('testing')
        # generator
        group.add_argument('--num_sample', type=int, default=10, help="Number test samples to use, -1 for all")
        group.add_argument('--gen_samples_test', type=int, default=10, help="Number of completion outputs per sample")

        # contrastive and gaussian process
        group.add_argument('--grid_size', type=int, default=100, help="Resolution in each dim for predictive grid")
        group.add_argument('--noise_variance', type=float, default=0.01, help="Noise in observed partial cloud")
        group.add_argument('--gp_batch', type=int, default=1000, help="Batch size for Gaussian posterior prediction")


if __name__ == '__main__':
    pass
