import torch
from torch.nn import Parameter
# import torch.autograd as autograd
import torch.nn.functional as F


def gen_nearest_latents(dci_db, gen_data, complete_data):
    indices = []
    gen_data_sampled = []

    for s in range(gen_data.shape[0]):
        gen_sample = gen_data[s]
        complete_sample = torch.unsqueeze(complete_data[s], 0)
        # try: (adding data)
        dci_db.add(gen_sample)
        # indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
        index, _ = dci_db.query(complete_sample, 1, 5000)
        indices.append(index)
        dci_db.clear()
        gen_data_sampled.append(gen_sample[index[0][0].long()])

    # dci_db.free()
    gen_data = torch.stack(gen_data_sampled)
    torch.cuda.empty_cache()

    return gen_data


def gen_nearest_latents_with_indices(dci_db, gen_data, complete_data):
    indices = []
    gen_data_sampled = []

    for s in range(gen_data.shape[0]):
        gen_sample = gen_data[s]
        complete_sample = torch.unsqueeze(complete_data[s], 0)
        # try: (adding data)
        dci_db.add(gen_sample)
        # indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
        index, _ = dci_db.query(complete_sample, 1, 5000)
        indices.append(index[0][0].long())
        dci_db.clear()
        gen_data_sampled.append(gen_sample[index[0][0].long()])

    # dci_db.free()
    stacked_indices = torch.stack(indices)
    gen_data = torch.stack(gen_data_sampled)
    torch.cuda.empty_cache()

    return gen_data, stacked_indices


def get_nearest_mapping(dci_db, manifold_predictions, zeros, z_data):
    indices = []
    manifold_pred_selected = []
    z_data_used = []

    for s in range(manifold_predictions.shape[0]):
        manifold_pred = manifold_predictions[s]
        z_sample = z_data[s]
        # try: (adding data)
        dci_db.add(manifold_pred)
        # indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
        index, _ = dci_db.query(zeros, 1, 5000)
        indices.append(index[0][0].long())
        dci_db.clear()
        manifold_pred_selected.append(manifold_pred[index[0][0].long()])
        z_data_used.append(z_sample[index[0][0].long()])

    # dci_db.free()
    manifold_pred_data = torch.stack(manifold_pred_selected)
    z_data = torch.stack(z_data_used)
    torch.cuda.empty_cache()

    return manifold_pred_data, z_data


def _weight_drop(module, weights, device, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module.to(device), 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, device='cpu', dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, device, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, device, dropout)
        self.forward = module.forward


def coordinate2index(x, reso, coord_type='2d'):
    """
        Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

        Args:
            x (tensor): coordinate
            reso (int): defined resolution
            coord_type (str): coordinate type
    """
    x = (x * reso).long()
    index = None
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def normalize_coordinate(p, plane='xz',bbox_size=1.0):
    """
        Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            plane (str): plane feature type, ['xz', 'xy', 'yz']
            bbox_size: box size in units, default-1
    """
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / bbox_size # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, bbox_size=1.0):
    """
        Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

        Args:
            p (tensor): point
            bbox_size: box size in units, default-1
    """
    p_nor = p / bbox_size # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def depth_to_volume(x, upsample_ratio):
    N, C, D, H, W = x.shape
    x = x.view(N, upsample_ratio, upsample_ratio, upsample_ratio, C //
               (upsample_ratio ** 3), D, H, W)  # (N, bs, bs, bs, C//bs^3, D, H, W)
    # (N, C//bs^2, D, bs H, bs, W, bs)
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
    x = x.view(N, C // (upsample_ratio ** 3), D * upsample_ratio, H *
               upsample_ratio, W * upsample_ratio)  # (N, C//bs^2, H * bs, W * bs)
    return x


def volume_to_depth(x, upsample_ratio):
    N, C, D, H, W = x.shape
    x = x.view(N, C, D // upsample_ratio, upsample_ratio, H // upsample_ratio, upsample_ratio,
               W // upsample_ratio, upsample_ratio)  # (N, C, D//bs, bs, H//bs, bs, W//bs, bs)
    # (N, bs, bs, bs, C, D//bs, H//bs, W//bs)
    x = x.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()
    x = x.view(N, C * (upsample_ratio ** 3), H // upsample_ratio,
               W // upsample_ratio)  # (N, C*bs^2, H//bs, W//bs)
    return x

def grid_sample_2d(input_tensor, grid):
    # grid (B, Ho, Wo, 2)
    # input (B, C, H, W)
    # output (B, C, Ho, Wo)
    B, Ho, Wo, _ = grid.shape
    P = Ho * Wo
    _, C, H, W = input_tensor.shape

    # grid to index
    reso = torch.tensor([2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 2).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    w1h1 = torch.floor(grid).long()
    w2h2 = w1h1 + 1
    assert(w1h1.min() >= 0 and w2h2[..., 0].max() <= W and w2h2[..., 1].max() <= H)

    input_tensor = F.pad(input_tensor, (0, 1, 0, 1), mode='replicate')

    # B, 4*P: Q11, Q12, Q21, Q22
    index = torch.stack([w1h1[...,1]*(W+1)+w1h1[...,0], (w1h1[...,1]+1)*(W+1)+w1h1[...,0],
                            w1h1[...,1]*(W+1)+w1h1[...,0]+1, (w1h1[...,1]+1)*(W+1)+w1h1[...,0]+1], dim=1).view(B, -1)
    # 4 of (B, C, P) (f11, f12, f21, f22)
    input_4samples = torch.gather(input_tensor.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P, dim=-1)

    # B, Ho, Wo, 2 -> B, P
    diff_x2x, diff_y2y = torch.unbind((w2h2 - grid).view(B, -1, 2), dim=-1)
    diff_xx1, diff_yy1 = torch.unbind((grid - w1h1).view(B, -1, 2), dim=-1)

    f11, f12, f21, f22 = input_4samples
    # B,1,P,2 @ B,C,P,2,2
    result = torch.stack([diff_x2x, diff_xx1],dim=-1).reshape(B,1,P,1,2).expand(-1,C,-1,-1,-1).reshape(-1,1,2) @ torch.stack([f11, f12, f21, f22],dim=-1).reshape(B,C,P,2,2).reshape(-1,2,2) @ torch.stack([diff_y2y, diff_yy1],dim=-1).reshape(B,1,P,2,1).expand(-1,C,-1,-1,-1).reshape(-1,2,1)
    result = result.view(B,C,Ho,Wo)

    return result

def grid_sample_3d(input_tensor, grid):
    """
    grid (B, Do, Ho, Wo, 3)
    input (B, C, D, H, W)
    output (B, C, Do, Ho, Wo)
    """
    B, Do, Ho, Wo, _ = grid.shape
    P = Ho * Wo * Do
    _, C, D, H, W = input_tensor.shape

    # ref = F.grid_sample(input, grid, padding_mode="border", align_corners=True)
    # grid to index B,1,1,1,3
    reso = torch.tensor([2.0/ (D-1), 2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 1, 3).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    x000 = torch.floor(grid).long()

    input_tensor = F.pad(input_tensor, (0, 1, 0, 1, 0, 1), mode='replicate')

    # B, 8*P: Q000, Q001, Q010, Q011, Q100, Q101, Q110, Q111
    index = torch.stack([(x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         ], dim=1).view(B, -1)
    # 2 of (B, C, 4P) corresponding f000, f001, f010, f011, f100, f101, f110, f111
    f0xx, f1xx = torch.gather(input_tensor.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P*4, dim=-1)

    # B, Ho, Wo, 3 -> B, P
    xd, yd, zd = torch.unbind((grid - x000).view(B, -1, 3), dim=-1)

    # B, C, 4P
    fxx = f0xx * (1-xd).repeat(1, 4).unsqueeze(1) + f1xx * xd.repeat(1,4).unsqueeze(1)

    # f00, f01: B, C, 2P
    f0x, f1x = fxx.split(2*P, dim=-1)
    fx = f0x * (1-yd).repeat(1, 2).unsqueeze(1) + f1x*yd.repeat(1,2).unsqueeze(1)

    f0, f1 = fx.split(P, dim=-1)
    result = f0*(1-zd).unsqueeze(1) + f1*zd.unsqueeze(1)

    result = result.view(B,C,Do,Ho,Wo)
    return result
