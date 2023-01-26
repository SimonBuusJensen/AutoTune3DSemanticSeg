import argparse
import os
import random
import glob
import torch

from tqdm import tqdm
import numpy as np
import pandas as pd

"""
Considerations regarding Voxel size:
- Memory at test time: 
    The voxel size has to be set, such that the largest point cloud can fit into a GPUs memory during test/inference time.
    - How much memory does PointNeXt take up per point?
- Density: 
    The voxel size decreases the density of the point cloud. 
    The voxel size should not be too great, such that the density of the point clouds is too low. 
    It is too low, if there are no points left in the latter set abstraction layers of PointNeXt.
- Number of points: 
    The voxel size decreases the number of points in the point clouds. 
    The voxel size should not be too great, such that there are less points left in the point cloud 
    than what the voxel max hyper-parameter is set to.
"""


def get_gpu_info(verbose=True):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved

    if verbose:
        print('GPU Info:')
        # Print the name of the GPU
        print("GPU name: ", torch.cuda.get_device_name(0))

        # Print memory in Gigabytes
        print(f"Total memory: {t / 1024 ** 3} GB")
        print(f"Reserved memory: {r / 1024 ** 3} GB")
        print(f"Allocated memory: {a / 1024 ** 3} GB")
        print(f"Free memory: {f / 1024 ** 3} GB")
        print('')

    return t, f


def init_empty_pcld_dimension_df():
    """
    Initialize an empty dataframe which will hold dimensions of the point clouds
    """
    df = pd.DataFrame(columns=['file_name', 'n_points', 'x_max', 'y_max', 'z_max', 'volume'])
    return df


def set_initial_voxel_size(file_names):
    # Create a dataframe which holds the dimensions of the point clouds
    df = init_empty_pcld_dimension_df()
    file_names = tqdm(file_names)
    for file_name in file_names:
        # Get the full path to the file
        file_path = os.path.join(args.data_root, file_name)

        # Read the file
        data = np.load(file_path)
        coords = data[:, coord_cols]

        # Calculate the number of points
        n_points = coords.shape[0]

        # Axis align the point cloud around the origin (0, 0, 0)
        coords -= coords.min(axis=0)

        # Calculate the min and max values of the coordinates
        x_max = np.max(coords[:, 0])
        y_max = np.max(coords[:, 1])
        z_max = np.max(coords[:, 2])
        volume = x_max * y_max * z_max

        # Input into the dataframe
        df = pd.concat([df, pd.DataFrame([[file_name,
                                           n_points,
                                           x_max,
                                           y_max,
                                           z_max,
                                           volume
                                           ]],
                                         columns=df.columns)], ignore_index=True)

    df.to_csv("volumes.csv", index=False)
    pcld_dimensions_df = pd.read_csv("volumes.csv")
    mean_pcld_volume = np.mean(pcld_dimensions_df['volume'])

    # Calculate the voxel size based on the mean point cloud volume and the ratio
    voxel_size = pow(mean_pcld_volume / voxel_size_2_pcld_ratio, (1 / 3))

    return voxel_size


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
                 np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def find_voxelized_point_count(points, voxel_size):
    """
    Find the max number of points in a voxel downsampled point cloud
    Returns: The number of points in a voxel down-sampled point cloud
    """
    discrete_coord = np.floor(points / np.array(voxel_size))

    key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)

    return count.shape[0]

    # if mode == 0:  # train mode
    #     idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + \
    #                  np.random.randint(0, count.max(), count.size) % count
    #     idx_unique = idx_sort[idx_select]
    #     return idx_unique
    # else:  # val mode
    #     return idx_sort, voxel_idx, count


if __name__ == '__main__':

    # Argparse for reading a data_root
    parser = argparse.ArgumentParser(
        'Data analyzer for hyper-parameter tuning of PointNeXt using the mean volume of a point cloud dataset')
    parser.add_argument('--data_root', type=str, required=False,
                        help='data root (expects a folder of .npy files with point clouds)',
                        # default="/home/ambolt/data/inropa/dataset/point_clouds")
                        # default="/home/ambolt/data/s3dis/stanford_indoor3d/raw")
                        # default="/home/ambolt/data/Novafos3D")
                        default="/home/ambolt/data/semantic-kitti/dataset")
    parser.add_argument('--coord_cols', nargs='+', type=int,
                        help='columns of the coordinates in the point clouds (x, y, z)',
                        default=[0, 1, 2])
    parser.add_argument('--pre_normalize', type=bool,
                        help="whether to scale the dimension of the point clouds to between 0-1",
                        default=False)
    parser.add_argument('--voxel_max', type=int, help="Points which will be sampled from the point cloud",
                        default=24000)
    parser.add_argument('--voxel_size_2_pcld_ratio', type=float,
                        help='ratio of the voxel size to the raw point cloud sizes',
                        default=1_275_156)
    parser.add_argument('--max_point_clouds', type=int, default=1000, help="Maximum number of points clouds to process")

    args = parser.parse_args()
    data_root = args.data_root
    coord_cols = args.coord_cols
    voxel_max = args.voxel_max
    voxel_size_2_pcld_ratio = args.voxel_size_2_pcld_ratio
    max_point_clouds = args.max_point_clouds

    """ 
    1. Identify GPU memory available in GBs
    """
    total_mem, free_mem = get_gpu_info()

    """
    2. Calculate volume of point clouds and count number of points and save it to a csv file
    """
    # Get the file names from the data_root
    file_names = []
    if data_root.split("/")[-2] == "semantic-kitti":
        for sequence in range(0, 11):
            sequence_str = str(sequence).zfill(2)
            sequence_root = os.path.join(data_root, sequence_str)
            file_names = file_names + [os.path.join(sequence_str, fn) for fn in glob.glob1(sequence_root, "*.npy")]
        # select 1000 samples from the dataset randomly
        random.seed(42)
        file_names = random.sample(file_names, max_point_clouds)
    else:
        file_names = sorted(glob.glob1(data_root, "*.npy"))

    df = pd.DataFrame(columns=['file_name', 'n_points', 'x_max', 'y_max', 'z_max', 'volume'])
    file_names = tqdm(file_names)
    max_num_points = 0
    max_num_points_file_name = ""
    for file_name in file_names:
        # Get the full path to the file
        file_path = os.path.join(data_root, file_name)

        # Read the file
        data = np.load(file_path)

        # How many points?
        num_points = data.shape[0]

        volume = data[:, coord_cols].max(axis=0).prod()

        max_x = data[:, coord_cols[0]].max()
        max_y = data[:, coord_cols[1]].max()
        max_z = data[:, coord_cols[2]].max()

        # Save the results to a dataframe
        df = pd.concat([df,
                        pd.DataFrame([[file_name, num_points, max_x, max_y, max_z, volume]], columns=df.columns)],
                       ignore_index=True)
        df.to_csv(".csv", index=False)


    # Print the maximum number of points and the name of the file
    print(f"File name: {max_num_points_file_name}")
    print(f"Maximum number of points: {max_num_points}")

    """
    3. If the point clouds with the most points have less points than voxel max don't perform voxel down-sampling
    """
    if not max_num_points > voxel_max:
        print("No voxel down-sampling required")
        exit(0)

    """
    4. Find voxel size which is appropriate given the voxel max and the GPU memory
    """
    data = np.load(os.path.join(data_root, max_num_points_file_name))

    voxelized_pcld_size = 0
    voxel_size = None
    while voxelized_pcld_size <= voxel_max:
        # find initial voxel size given the ratio from S3DIS
        if voxel_size is None:
            voxel_size = set_initial_voxel_size(file_names)
        else:
            # Decrease the voxel size by 10% until the voxelized point cloud size is greater than the voxel max
            voxel_size = voxel_size * 0.9

        voxelized_pcld_size = find_voxelized_point_count(data[:, coord_cols], voxel_size)

    print(voxel_size, voxelized_pcld_size)
    """ 
    Find the model size of PointNeXt per point
    """
    # print('Number of params: %.4f M' % (model_size / 1e6))
