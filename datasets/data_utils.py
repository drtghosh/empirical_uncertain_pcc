import os
import numpy as np
from scipy import spatial
from chainer.backends import cuda
from joblib import Parallel, delayed
import torch
import h5py
import laspy as lp
import open3d as o3d
from gpytoolbox import apply_colormap, colormap
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
    random_distance = np.random.uniform(0.05, 0.5, len(points))
    # random_distance[np.abs(random_distance) < 0.01] = np.random.choice([-1, 1])
    for p in range(len(normals)):
        negative_data[p] = points[p] + random_distance[p] * normals[p]

    return negative_data


def create_negative_on_grid(path, grid_size=128, eps=0.1, support=0.02):
    point_cloud = o3d.io.read_point_cloud(path)
    points = np.array(point_cloud.points)
    # create array of grid sizes
    space_dim = points.shape[-1]
    grid_sizes = np.ones(space_dim, dtype=np.int32) * grid_size
    # find the bounding box for all dataset
    box_min = np.amin(points, 0) - eps
    box_max = np.amax(points, 0) + eps

    # Build a grid (dimension-agnostic)
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(space_dim)])
    grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, space_dim)
    # grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
    kdtree = spatial.KDTree(points)
    indices_to_remove = []
    for i, v in enumerate(grid_vertices):
        distance, index = kdtree.query(v)
        if distance < support:
            indices_to_remove.append(i)
    grid_vertices_new = np.delete(grid_vertices, indices_to_remove, 0)
    grid_vertices_save = grid_vertices_new[np.random.permutation(len(grid_vertices_new))[:len(points)]]

    return grid_vertices_save


def save_negative_complete_pcn(data_root, split, category, cat2id, use_normal=False, on_grid=False):
    negative_folder = 'negative'
    assert split in ['train', 'validation', 'test'], "split error value!"
    with open(os.path.join(data_root, split + '.list'), 'r') as f:
        lines = f.read().splitlines()

    cat_id = cat2id[category]
    lines = list(filter(lambda x: x.startswith(cat_id), lines))

    if use_normal:
        negative_folder = 'negative_from_normal'

    if on_grid:
        negative_folder = 'negative_grid'

    # noinspection PyTypeChecker
    negative_cat_path = os.path.join(data_root, split, negative_folder, cat_id)
    if not os.path.exists(negative_cat_path):
        os.makedirs(negative_cat_path)

    for line in lines:
        new_path = os.path.join(data_root, split, negative_folder, line + '.ply')
        if os.path.exists(new_path):
            pass
        else:
            file_path = os.path.join(data_root, split, 'complete', line + '.ply')
            if use_normal:
                new_points = create_negative_with_normal(file_path)
            elif on_grid:
                new_points = create_negative_on_grid(file_path)
            else:
                new_points = create_negative_data(file_path)
            new_pc = o3d.geometry.PointCloud()
            new_pc.points = o3d.utility.Vector3dVector(new_points)
            o3d.io.write_point_cloud(new_path, new_pc)


def save_negative_complete_pcn_parallel(data_root, split, category, cat2id, use_normal=False, on_grid=False, jobs=6):
    negative_folder = 'negative'
    assert split in ['train', 'validation', 'test'], "split error value!"
    with open(os.path.join(data_root, split + '.list'), 'r') as f:
        lines = f.read().splitlines()

    cat_id = cat2id[category]
    lines = list(filter(lambda x: x.startswith(cat_id), lines))

    if use_normal:
        negative_folder = 'negative_from_normal'

    if on_grid:
        negative_folder = 'negative_grid'

    # noinspection PyTypeChecker
    negative_cat_path = os.path.join(data_root, split, negative_folder, cat_id)
    if not os.path.exists(negative_cat_path):
        os.makedirs(negative_cat_path)

    # parallelization
    Parallel(n_jobs=jobs)(
        delayed(save_negative_complete_pcn_file)(data_root, split, negative_folder, line, use_normal, on_grid) for line
        in lines)


def save_negative_complete_pcn_file(data_root, split, negative_folder, line, use_normal=False, on_grid=False):
    new_path = os.path.join(data_root, split, negative_folder, line + '.ply')
    if os.path.exists(new_path):
        pass
    else:
        file_path = os.path.join(data_root, split, 'complete', line + '.ply')
        if use_normal:
            new_points = create_negative_with_normal(file_path)
        elif on_grid:
            new_points = create_negative_on_grid(file_path)
        else:
            new_points = create_negative_data(file_path)
        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(new_points)
        o3d.io.write_point_cloud(new_path, new_pc)


id_dict = {
    # seen categories
    "airplane": "02691156",  # plane
    "cabinet": "02933112",  # dresser
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "vessel": "04530566",  # boat
}
# save_negative_complete_pcn_parallel('data/PCN', 'train', 'table', id_dict, False, True, 12)
# save_negative_complete_pcn_parallel('data/PCN', 'validation', 'table', id_dict, False, True)


def create_grid(test_data, grid_size, space_dim=3, box_min=None, box_max=None, eps=0.2):
    # send data to cpu
    test_data = test_data.cpu()
    # create array of grid sizes
    grid_sizes = np.ones(space_dim, dtype=np.int32) * grid_size
    # find the bounding box for all dataset
    if box_min is None:
        box_min = torch.amin(test_data, 1)[0] - eps
    if box_max is None:
        box_max = torch.amax(test_data, 1)[0] + eps
    # compute the grid spacing
    grid_spacing = (box_max - box_min) / (grid_size - 1)

    # Build a grid (dimension-agnostic)
    grid_vertices = np.meshgrid(
        *[np.linspace(box_min[d], box_max[d], grid_sizes[d]) for d in range(space_dim)])
    grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, space_dim)
    grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
    return grid_vertices.unsqueeze(0), grid_sizes, box_min.numpy(), grid_spacing.numpy()


def create_negative_with_label(point_cloud_tensor, distance=1):
    if len(point_cloud_tensor.size()) > 2:
        point_cloud_tensor = point_cloud_tensor[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_tensor.cpu().numpy())
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
    selected_gen = np.random.randint(2, len(pcds)-2, 4)
    selected_gen = np.concatenate(([0, 1], [len(pcds) - 2, len(pcds) - 1], selected_gen))
    pcds = [pcds[i] for i in selected_gen]
    titles = np.array(titles)[selected_gen]
    colors = [colors[i] for i in selected_gen]
    if sizes is None:
        sizes = [0.5] * len(pcds)
    fig = plt.figure(figsize=((len(pcds) // 2) * 3 * 1.4, 2 * 3 * 1.4))
    # fig, axes = plt.subplots(2, 4)
    elev = 30
    azim = -45
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        # color = pcd[:, 0]
        ax = fig.add_subplot(2, len(pcds) // 2, j + 1, projection='3d')
        ax.view_init(elev, azim)
        # axes[j // 4, j % 4].scatter(..., cmap='viridis', vmin=-1.0, vmax=0.5)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=colors[j], s=size)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def write_ply(filename, vertices, faces=None, colors=None, cmap='BuGn'):
    """Store triangle mesh into .ply file format

    Writes a triangle mesh (optionally with per-vertex colors) into the ply file format, in a way consistent with, e.g., importing to Blender.

    Parameters
    ----------
    filename : str
        Name of the file ending in ".ply" to which to write the mesh
    vertices : numpy double array
        Matrix of mesh vertex coordinates
    faces : numpy int array, optional (default None)
        Matrix of triangle face indices into vertices. If none, only the vertices will be written (e.g., a point cloud)
    colors : numpy double array, optional (default None)
        Array of per-vertex colors. It can be a matrix of per-row RGB values, or a vector of scalar values that gets transformed by a colormap.
    cmap : str, optional (default 'BuGn')
        Name of colormap used to transform the color values if they are a vector of scalar function values (if colors is a matrix of RGB values, this parameter will not be used). Should be a valid input to `colormap`.

    See Also
    --------
    write_mesh, colormap.

    Notes
    -----
    This function is not optimized and covers the very specific funcionality of saving a mesh with per-vertex coloring that can be imported into Blender or other software. If you wish to write a mesh for any other purpose, we strongly recommend you use write_mesh instead.

    Examples
    --------
    TODO
    """

    vertices = vertices.astype(float)
    f = open(filename, "w")
    f.write("ply\nformat {} 1.0\n".format('ascii'))
    f.write("element vertex {}\n".format(vertices.shape[0]))
    f.write("property double x\n")
    f.write("property double y\n")
    f.write("property double z\n")
    if colors is not None:
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
    if faces is not None:
        f.write("element face {}\n".format(faces.shape[0]))
    else:
        f.write("element face 0\n")
    f.write("property list int int vertex_indices\n")
    f.write("end_header\n")
    # write_vert_str = "{} {} {}\n" * vertices.shape[0]
    # f.write(write_vert_str.format(tuple(np.reshape(vertices,(-1,1)))))
    # This for loop should be vectorized
    if colors is None:
        for i in range(vertices.shape[0]):
            f.write("{} {} {}\n".format(vertices[i,0],vertices[i,1],vertices[i,2]))
    else:
        if colors.ndim == 1 or colors.shape[1] == 1:  # color is scalar values
            C = apply_colormap(colormap(cmap, 200), colors)
        else:
            if np.max(colors) <= 1:
                C = np.round(colors*255)
            else:
                C = colors
        # This should be vectorized
        for i in range(vertices.shape[0]):
            f.write("{} {} {} {} {} {} 255\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2], int(C[i, 0]),
                                                     int(C[i, 1]), int(C[i, 2])))
    # This should be vectorized
    if faces is not None:
        for i in range(faces.shape[0]):
            f.write("3 {} {} {}\n".format(faces[i, 0], faces[i, 1], faces[i, 2]))
    f.close()
