import os
import numpy as np
from chainer.backends import cuda
import torch
import h5py
import laspy as lp
import open3d as o3d
from matplotlib import pyplot as plt


def read_point_cloud_ply(path):
    pc = o3d.io.read_point_cloud(path)
    return np.array(pc.points, np.float32)


def write_point_cloud_ply(points, path, with_color=False, colors=None):
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if with_color:
        if torch.is_tensor(colors):
            colors = colors.cpu().detach().numpy()
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pc)


def read_point_cloud_las(path):
    pc = lp.read(path)
    return np.array(pc.xyz, np.float32)


def read_point_cloud_npy(path):
    return np.load(path)


def read_point_cloud_h5(path):
    f = h5py.File(path, 'r')
    return f['data'][()]


def add_noise_pc(pc, sigma=0.01):
    n = pc.shape[0]
    for p in range(n):
        pc[p][0] += sigma * np.random.randn()
        pc[p][1] += sigma * np.random.randn()
        pc[p][2] += sigma * np.random.randn()

    return pc


def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if pc.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
    return pc[idx[:n]]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start the farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th` farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """

    pts = pts[np.newaxis, :, :]

    ndim = pts.shape[2]
    if ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k,), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    min_distances = None
    # noinspection PyBroadException
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception:
        import IPython
        IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)

    pts = pts[:, indices, :]
    return pts[0][0]


def positional_encoding(tensor, encoding_size=6, include_input=True, log_sampling=True) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether to include the input in the
            positional encoding (default: True).
        log_sampling (optional, bool): Whether to use log sampling
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            encoding_size - 1,
            encoding_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (encoding_size - 1),
            encoding_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def create_negative_data(path):
    point_cloud = o3d.io.read_point_cloud(path)
    points = np.array(point_cloud.points)

    mid = (np.max(points, 0) - np.min(points, 0)) / 2
    avg_distance = np.mean(mid) * np.random.uniform(0.01, 0.2, (points.shape[0], 1))
    noise = 0.01 * np.random.randn(points.shape[0], 2)

    bias = np.concat((avg_distance, noise), axis=1)
    bias_tensor = torch.tensor(bias)
    indices = torch.argsort(torch.rand(*bias_tensor.shape), dim=-1)
    bias_tensor = bias_tensor[torch.arange(bias_tensor.shape[0]).unsqueeze(-1), indices]
    new_points = points + bias_tensor.numpy()

    return new_points


def create_negative_with_normal(path):
    point_cloud = o3d.io.read_point_cloud(path)
    point_cloud.estimate_normals()
    points = np.array(point_cloud.points)
    normals = np.array(point_cloud.normals)
    negative_data = np.empty(points.shape)
    random_distance = np.random.randn(len(points))
    random_distance[np.abs(random_distance) < 0.01] = np.random.choice([-1, 1])
    for p in range(len(normals)):
        negative_data[p] = points[p] + random_distance[p] * normals[p]

    return negative_data


def save_negative_complete_pcn(data_root, split, category, cat2id, use_normal=False):
    assert split in ['train', 'validation', 'test'], "split error value!"
    with open(os.path.join(data_root, split + '.list'), 'r') as f:
        lines = f.read().splitlines()

    cat_id = cat2id[category]
    lines = list(filter(lambda x: x.startswith(cat_id), lines))

    for line in lines:
        file_path = os.path.join(data_root, split, 'complete', line + '.ply')
        if use_normal:
            new_points = create_negative_with_normal(file_path)
        else:
            new_points = create_negative_data(file_path)
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(new_points)
        # noinspection PyTypeChecker
        negative_cat_path = os.path.join(data_root, split, 'negative', cat_id)
        if not os.path.exists(negative_cat_path):
            os.makedirs(negative_cat_path)
        o3d.io.write_point_cloud(os.path.join(data_root, split, 'negative', line + '.ply'), new_pc)


id_dict = {
    # seen categories
    "airplane": "02691156",  # plane
    "cabinet": "02933112",  # dresser
    "car": "02958343"
}
# save_negative_complete_pcn('data/PCN', 'validation', 'car', id_dict, True)


def create_grid(test_data, grid_size, space_dim=3, box_min=None, box_max=None, eps=0.2):
    # create array of grid sizes
    grid_sizes = np.ones(space_dim, dtype=np.int32) * grid_size
    # find the bounding box for all dataset
    if box_min is None:
        box_min = torch.amin(test_data, 1)[0] - eps
    if box_max is None:
        box_max = torch.amax(test_data, 1)[0] + eps

    # Build a grid (dimension-agnostic)
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(space_dim)])
    grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, space_dim)
    grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
    return grid_vertices.unsqueeze(0), grid_sizes


def create_negative_with_label(point_cloud, distance=1):
    if len(point_cloud.size()) > 2:
        point_cloud = point_cloud[0]
    point_cloud.estimate_normals()
    points = np.array(point_cloud.points)
    normals = np.array(point_cloud.normals)
    negative_data = np.empty(points.shape)
    random_distance = distance * np.random.randn(len(points))
    for p in range(len(normals)):
        negative_data[p] = points[p] + random_distance[p] * normals[p]
    negative_data = torch.tensor(negative_data, dtype=torch.float32).unsqueeze(0)
    negative_label = torch.tensor(random_distance, dtype=torch.float32)
    return negative_data, negative_label


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, colors=None, zdir='y', xlim=(-0.5, 0.5),
                      ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    selected_gen = np.random.randint(2, len(pcds)-2, 3)
    selected_gen = np.concatenate(([0, 1], selected_gen, [len(pcds) - 1]))
    pcds = [pcds[i] for i in selected_gen]
    titles = np.array(titles)[selected_gen]
    colors = [colors[i] for i in selected_gen]
    if sizes is None:
        sizes = [0.5] * len(pcds)
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30
    azim = -45
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        # color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=colors[j], s=size, cmap='viridis', vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
