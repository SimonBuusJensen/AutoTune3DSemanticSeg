import os
import argparse
import glob
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Version 2: allows pre-normalization of the point clouds based on the max dimension found in the entire dataset
"""


def init_empty_pcld_dimension_df():
    """
    Initialize an empty dataframe which will hold dimensions of the point clouds
    """
    df = pd.DataFrame(columns=['file_name', 'n_points', 'x_max', 'y_max', 'z_max', 'volume'])
    return df


def init_empty_voxelized_pcld_dimension_df():
    """
    Initialize an empty dataframe which will hold dimensions of the voxel-downsampled and cropped point clouds
    """
    df = pd.DataFrame(columns=['file_name', 'n_points', 'x_max', 'y_max', 'z_max', 'volume'])
    return df


def set_initial_voxel_size(args, df_name="point_cloud_dimensions.csv"):
    """
    Find the optimal voxel size for the point clouds in the dataset.
    """

    # Good ratio between the voxel size and the point cloud volumes as used for S3DIS in PointNeXt paper
    voxel_size_2_pcld_ratio = args.voxel_size_2_pcld_ratio

    """
        Fill the dataframe with the dimensions of the point clouds.
        Iterate over the point clouds and calculate the dimensions of the point clouds.
        """

    # Get the file names from the data_root
    file_names = []
    if args.data_root.split("/")[-2] == "semantic-kitti":
        for sequence in range(0, 11):
            sequence_str = str(sequence).zfill(2)
            sequence_root = os.path.join(args.data_root, sequence_str)
            file_names = file_names + [os.path.join(sequence_str, fn) for fn in glob.glob1(sequence_root, "*.npy")]
        # select 1000 samples from the dataset randomly
        random.seed(42)
        file_names = random.sample(file_names, args.max_point_clouds)
    else:
        file_names = sorted(glob.glob1(args.data_root, "*.npy"))

    # Wrap file_names in tqdm for progress bar
    coord_cols = args.coord_cols

    # Create a dataframe which holds the dimensions of the point clouds
    df = init_empty_pcld_dimension_df()

    # Iterate over the point clouds and calculate number of points and dimensions
    print(f"Calculating the dimensions of the point clouds, which can be found in {df_name}...")
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

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    df.to_csv(os.path.join(args.out_dir, df_name), index=False)

    pcld_dimensions_df = pd.read_csv(os.path.join(args.out_dir, df_name))

    mean_pcld_volume = np.mean(pcld_dimensions_df['volume'])
    mean_pcld_n_points = np.mean(pcld_dimensions_df['n_points'])

    # Calculate the voxel size based on the mean point cloud volume and the ratio
    if mean_pcld_n_points > args.k:
        voxel_size = (mean_pcld_volume / voxel_size_2_pcld_ratio) ** (1 / 3)
    else:
        print(
            f"No voxel size is needed. The average point cloud has less points than k. {mean_pcld_n_points} < {args.k}")
        voxel_size = None

    # return the voxel size
    return voxel_size


def hyper_param_tune_voxel_size(args, voxel_size, df_name="voxelized_point_cloud_dimensions.csv"):
    """
    Find the optimal voxel size for the point clouds in the dataset.
    """
    k_neighbors = args.k

    """
    Fill the dataframe with the dimensions of the voxel-downsampled and cropped point clouds.
    Iterate over the point clouds:
     1. voxel-down sample the point cloud
     2. Crop the point cloud
     3. Calculate the dimensions of the cropped point cloud
    """
    # Get the file names from the data_root
    file_names = []
    if args.data_root.split("/")[-2] == "semantic-kitti":
        for sequence in range(0, 11):
            sequence_str = str(sequence).zfill(2)
            sequence_root = os.path.join(args.data_root, sequence_str)
            file_names = file_names + [os.path.join(sequence_str, fn) for fn in glob.glob1(sequence_root, "*.npy")]
        # select 1000 samples from the dataset randomly
        random.seed(42)
        file_names = random.sample(file_names, args.max_point_clouds)
    else:
        file_names = sorted(glob.glob1(args.data_root, "*.npy"))

    coord_cols = args.coord_cols

    # Make output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Iterate over the point clouds and calculate number of points and dimensions

    n_points_voxelized = 0
    while n_points_voxelized < k_neighbors and voxel_size is not None:

        # Create a dataframe which holds the dimensions of the point clouds
        df = init_empty_pcld_dimension_df()
        df_name = f"voxelized_point_cloud_dimensions_{voxel_size}.csv"
        print(f"Calculating the point counts of the voxelized point clouds, which can be found in {df_name}...")
        file_names = tqdm(file_names)
        for file_name in file_names:
            # Get the full path to the file
            file_path = os.path.join(args.data_root, file_name)

            # Read the file
            data = np.load(file_path)
            coords = data[:, coord_cols]

            # Calculate the number of points
            coords, _, _ = crop_point_cloud(coords, None, None,
                                            voxel_size=voxel_size,
                                            voxel_max=None)

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

            # Save the dataframe
            df.to_csv(os.path.join(args.out_dir, df_name), index=False)

        # Calculate mean points per voxelized point cloud
        n_points_voxelized = np.mean(df['n_points'])

        if n_points_voxelized < k_neighbors:
            # Decrease the voxel size by 10% until the voxelized point cloud size is greater than the voxel max
            print(
                f"Voxel size was too large, meaning the average number of points per voxel downsampled point cloud is less than {k_neighbors} ({n_points_voxelized}).")
            print(" Decreasing voxel size by 10%. New voxel size: ", voxel_size * 0.9)
            voxel_size *= 0.9

    return voxel_size


def hyper_param_tune_query_ball_radius(args, voxel_size, df_name="voxelized_cropped_point_cloud_dimensions.csv"):
    """
    Find a good radius of the query balls for the point clouds in the dataset.
    """

    # Good ratio between the query balls and the voxel down-sampled point cloud volume used for S3DIS in PointNeXt paper
    query_ball_2_pcld_ratio = args.qb_2_pcld_ratio

    """
    Fill the dataframe with the dimensions of the voxel-downsampled and cropped point clouds.
    Iterate over the point clouds:
     1. voxel-down sample the point cloud
     2. Crop the point cloud
     3. Calculate the dimensions of the cropped point cloud
    """
    # Get the file names from the data_root
    file_names = []
    if args.data_root.split("/")[-2] == "semantic-kitti":
        for sequence in range(0, 11):
            sequence_str = str(sequence).zfill(2)
            sequence_root = os.path.join(args.data_root, sequence_str)
            file_names = file_names + [os.path.join(sequence_str, fn) for fn in glob.glob1(sequence_root, "*.npy")]
        # select 1000 samples from the dataset randomly
        random.seed(42)
        file_names = random.sample(file_names, args.max_point_clouds)
    else:
        file_names = sorted(glob.glob1(args.data_root, "*.npy"))

    coord_cols = args.coord_cols

    # Make output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Iterate over the point clouds and calculate number of points and dimensions
    print(f"Calculating the dimensions of the voxelized point clouds, which can be found in {df_name}...")
    file_names = tqdm(file_names)

    # Create a dataframe which holds the dimensions of the point clouds
    df = init_empty_pcld_dimension_df()
    df_name = f"voxelized_cropped_point_cloud_dimensions_{round(voxel_size, 3)}.csv"

    for file_name in file_names:
        # Get the full path to the file
        file_path = os.path.join(args.data_root, file_name)

        # Read the file
        data = np.load(file_path)
        coords = data[:, coord_cols]

        # Calculate the number of points
        coords, _, _ = crop_point_cloud(coords, None, None,
                                        voxel_size=voxel_size,
                                        voxel_max=args.k)

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

        # Save the dataframe
        df.to_csv(os.path.join(args.out_dir, df_name), index=False)

    voxelized_pcld_dimensions_df = pd.read_csv(os.path.join(args.out_dir, df_name))

    # Calculate the mean voxel down-sampled and cropped point cloud volume
    mean_pcld_volume = np.mean(voxelized_pcld_dimensions_df['volume'])

    # Calculate the query ball radius based on the mean point cloud volume and the ratio
    qb_radius = pow(mean_pcld_volume / query_ball_2_pcld_ratio, (1 / 3))

    return qb_radius


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


def voxel_downsample(coord, voxel_size=0.05, mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))

    key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[
                               0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, voxel_idx, count


def crop_point_cloud(coord, feat, label, voxel_size, voxel_max, variable=True, neighborhood_sampling=True,
                     shuffle=True):
    # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer.
    coord -= coord.min(0)

    if voxel_size is not None:
        uniq_idx = voxel_downsample(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, \
            label[uniq_idx] if label is not None else None
    if voxel_max is not None:
        crop_idx = None
        N = len(coord)  # the number of points
        if N >= voxel_max:
            if neighborhood_sampling:
                init_idx = np.random.randint(N)
                crop_idx = np.argsort(
                    np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            else:
                crop_idx = np.random.choice(N, voxel_max, replace=False)
        elif not variable:
            # fill more points for non-variable case (batched data)
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])

        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[
            crop_idx] if label is not None else None
    coord -= coord.min(0)
    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None, label.astype(
        np.long) if label is not None else None


if __name__ == '__main__':
    # Argparse for reading a data_root
    parser = argparse.ArgumentParser(
        'Automatic hyper-parameter tuning of voxel size and query ball radius for a 3D point cloud dataset')
    parser.add_argument('--data_root', type=str, required=False,
                        help='data root (expects a folder of .npy files with point clouds)',
                        default="examples/s3dis/")
    parser.add_argument('--coord_cols', nargs='+', type=int,
                        help='columns of the coordinates in the point clouds (x, y, z)',
                        default=[0, 1, 2])
    parser.add_argument('--voxel_size_2_pcld_ratio', type=float,
                        help='ratio of the voxel size to the raw point cloud volumes',
                        default=1_275_156)
    parser.add_argument('--k', type=int, default=24000,
                        help="crop size: number of points sampled from the raw point cloud in nearest neighbor fashion")
    parser.add_argument('--qb_2_pcld_ratio', type=float,
                        help='ratio of the query ball radius to the voxelized and cropped point cloud sizes',
                        default=33_630)
    parser.add_argument('--out_dir', type=str, help="output directory for statistics",
                        default="statistics")
    parser.add_argument('--max_point_clouds', type=int, default=1000, help="Maximum number of points clouds to analyze")
    args = parser.parse_args()

    """
    Calculate the voxel size based on the mean volume of point clouds and the voxel size to point cloud volume ratio.
    The ratio is based on the mean volume of the point clouds in the S3DIS dataset 
    and the settings of the PointNeXt paper.
    """
    voxel_size = set_initial_voxel_size(args)
    print(f"Initial voxel size: {round(voxel_size, 4) if voxel_size < 1 else round(voxel_size, 1)}")
    voxel_size = hyper_param_tune_voxel_size(args, voxel_size)

    """
    Calculate the query ball radius based on the mean volume of voxel-downsampled and cropped point clouds and the
    query ball volume to point cloud volume ratio. 
    The ratio is based on the mean volume of the point clouds in the S3DIS dataset 
    and the settings of the PointNeXt paper.
    """
    query_ball_radius = hyper_param_tune_query_ball_radius(args, voxel_size)
    print()
    print(f"Final Voxel size: {round(voxel_size, 4) if voxel_size < 1 else round(voxel_size, 1)}")
    print(
        f"Final Query ball radius: {round(query_ball_radius, 4) if query_ball_radius < 1 else round(query_ball_radius, 1)}")
