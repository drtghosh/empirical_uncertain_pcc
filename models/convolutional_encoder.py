"""
Convolutional Feature Embedding
"""
from typing import Callable, Dict, List, Tuple, Union
import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max
from .generic_models import ResnetBlockFC
from .unet import UNet
from .unet3d import UNet3D
import numpy as np


def get_knn(vs: torch.Tensor, k: int, batch_idx: torch.Tensor) -> torch.Tensor:
    mat_square = torch.matmul(vs, vs.transpose(2, 1))
    diag = torch.diagonal(mat_square, dim1=1, dim2=2)
    diag = diag.unsqueeze(2).expand(mat_square.shape)
    dist_mat = (diag + diag.transpose(2, 1) - 2 * mat_square)
    _, index = dist_mat.topk(k + 1, dim=2, largest=False, sorted=True)
    index = index[:, :, 1:].view(-1, k) + batch_idx[:, None] * vs.shape[1]
    return index.flatten()


def extract_angles(vs: torch.Tensor, distance_k: torch.Tensor, vs_k: torch.Tensor) -> Union[
    Tuple[torch.Tensor, ...], List[torch.Tensor]]:
    proj = torch.einsum('nd,nkd->nk', vs, vs_k)
    cos_angles = torch.clamp(proj / distance_k, -1., 1.)
    proj = vs_k - vs[:, None, :] * proj[:, :, None]
    # moving same axis points
    ma = torch.abs(proj).sum(2) == 0
    num_points_to_replace = ma.sum().item()
    if num_points_to_replace:
        proj[ma] = torch.rand(num_points_to_replace,
                              vs.shape[1], device=ma.device)
    proj = proj / torch.norm(proj, p=2, dim=2)[:, :, None]
    angles = torch.acos(cos_angles)
    return angles, proj


def min_angles(dirs: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    ref = dirs[:, 0]
    all_cos = torch.einsum('nd,nkd->nk', ref, dirs)
    all_sin = torch.cross(ref.unsqueeze(
        1).expand(-1, dirs.shape[1], -1), dirs, dim=2)
    all_sin = torch.einsum('nd,nkd->nk', up, all_sin)
    all_angles = torch.atan2(all_sin, all_cos)
    all_angles[:, 0] = 0
    all_angles[all_angles < 0] = all_angles[all_angles < 0] + 2 * np.pi
    all_angles, indices = all_angles.sort(dim=1)
    indices = torch.argsort(indices, dim=1)
    all_angles_0 = 2 * np.pi - all_angles[:, -1]
    all_angles[:, 1:] = all_angles[:, 1:] - all_angles[:, :-1]
    all_angles[:, 0] = all_angles_0
    all_angles = torch.gather(all_angles, 1, indices)
    return all_angles


def extract_rotation_invariant_features(k: int) -> Tuple[Callable[[Union[torch.Tensor, np.array]], torch.Tensor], int]:
    batch_idx = None
    num_features = k * 3 + 1

    def get_input(xyz: torch.Tensor) -> Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        nonlocal batch_idx
        if batch_idx is None or len(batch_idx) != xyz.shape[0] * xyz.shape[1]:
            batch_idx, _ = torch.meshgrid(
                [torch.arange(xyz.shape[0]), torch.arange(xyz.shape[1])])
            batch_idx = batch_idx.flatten().to(xyz.device)
        return xyz.view(-1, 3), batch_idx

    def extract(base_vs: Union[torch.Tensor, np.array]):
        nonlocal num_features
        with torch.no_grad():
            if type(base_vs) is np.array:
                base_vs = torch.Tensor(base_vs)
            batch_size, num_pts = base_vs.shape[:2]
            vs, batch_idx = get_input(base_vs)
            knn = get_knn(base_vs, k, batch_idx)
            vs_k = vs[knn].view(-1, k, vs.shape[1])
            distance = torch.norm(vs, p=2, dim=1)
            vs_unit = vs / distance[:, None]
            distance_k = distance[knn].view(-1, k)
            angles, proj_unit = extract_angles(vs_unit, distance_k, vs_k)
            proj_min_angle = min_angles(proj_unit, vs_unit)
            fe = torch.cat([distance.unsqueeze(1), distance_k,
                            angles, proj_min_angle], dim=1)
        return fe.view(batch_size, num_pts, num_features)

    return extract, num_features


class ConvolutionalFeature(nn.Module):
    """
        PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

        Attributes:
            c_dim (int): dimension of latent code c
            dim (int): input points dimension
            hidden_dim (int): hidden dimension of the network
            scatter_type (str): feature aggregation when doing local pooling
            unet (bool): weather to use U-Net
            unet_kwargs (str): U-Net parameters
            unet3d (bool): weather to use 3D U-Net
            unet3d_kwargs (str): 3D U-Net parameters
            reso_plane (int): defined resolution for plane feature
            reso_grid (int): defined resolution for grid feature
            plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
            n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(self):
        super().__init__()

        # defaults from shapenet 3plane
        self.c_dim: int = 32
        self.dim: int = 3
        self.hidden_dim: int = 32
        self.scatter_type: str = 'max'
        self.unet: bool = False
        self.unet_kwargs: dict = dict(
            depth=4, merge_mode='concat', start_filts=32)
        self.unet3d: bool = True
        self.unet3d_kwargs: dict = {
            "num_levels": 4,
            "f_maps": 32,
            "in_channels": 32,
            "out_channels": 32
        }
        self.reso_plane: int = 64
        self.reso_grid: int = 64
        self.subpixel_upsampling: int = 1  # ratio for subpixel upsampling
        self.plane_type: List[str] = ['grid']
        self.n_blocks: int = 5
        self.sample_mode: str = "bilinear"
        self.input_normals: bool = False
        self.bbox_size: float = 2.0
        self.clusternet_feature: bool = False

        self._initialize()

    def _initialize(self):
        # shortcuts
        hidden_dim = self.hidden_dim
        dim = self.dim
        n_blocks = self.n_blocks
        c_dim = self.c_dim
        unet_kwargs = self.unet_kwargs
        unet = self.unet
        unet3d = self.unet3d

        if self.clusternet_feature:
            self.extractor, dim = extract_rotation_invariant_features(8)
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.nonlinear_layer = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim * (self.subpixel_upsampling ** 2),
                             in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            unet3d_kwargs = self.unet3d_kwargs.copy()
            unet3d_kwargs["out_channels"] *= (self.subpixel_upsampling ** 3)
            self.unet3d = UNet3D(**self.unet3d_kwargs)
        else:
            self.unet3d = None

        if self.subpixel_upsampling > 1:
            self.ps = torch.nn.PixelShuffle(self.subpixel_upsampling)

        if self.scatter_type == 'max':
            self.scatter = scatter_max
        elif self.scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        self.last_epoch = -1
