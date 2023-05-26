# Automatic Hyperparmeter Tuning for Semantic Segmentation of 3D Point Clouds
# Introduction:
Welcome to the 3D Point Cloud Analyzer! This tool allows you to analyze 3D point clouds stored in .npy files. Below is a quick usage guide to get you started with the tool.
This repository contains code for automatically finding appropriate hyper-parameters for 3D semantic segmentation models suchs as:
- [PointNet++](https://github.com/charlesq34/pointnet2)
- [RandLa-Net](https://github.com/QingyongHu/RandLA-Net) 
- [PointNeXt](https://github.com/guochengqian/PointNeXt)

Dataset specific Hyperparameters which will be outputted by the analyzer:
- Voxel size (for voxel downsampling)
- Query ball radius (for locating neighbouring points)

# How to use:

## Setup:
Before you start, make sure you have a compatible version of Python installed on your system. This project has been tested on Python 3.8.

then, clone the repository using the following command:

```
git clone git@github.com:SimonBuusJensen/AutoTune3DSemanticSeg.git 
cd AutoTune3DSemanticSeg
```

Install python dependencies:
```
pip install -r requirements.txt
```

## Usage:
To analyze the point clouds and get appropriate hyper-parameters, you need to run the main.py script from the command line with the following command:

``` 
python main.py --data_root [DIR_WITH_NPY_FILES]
```

Here, [DIR_WITH_NPY_FILES] is the directory containing your .npy files. You need to replace [DIR_WITH_NPY_FILES] with the path to your directory.

### Required format of point clouds:
The point cloud data should be stored in .npy files (NumPy array files). Each file represents a 3D point cloud with points stored as rows in a 2-dimensional array. The x-, y-, and z-coordinates of the points are expected to be in column 0, 1, and 2 respectively.
However, this can be changed using the --coord_cols option

```
python main.py --data_root [DIR_WITH_NPY_FILES] --coord_cols 2 3 4  # x: in column 2, y: in column 3, z: in column 4 
```

## Testing
The repository includes an examples/s3dis directory with example .npy files that you can use for testing.

To run the analyzer on these example files, you can use the following command:
``` 
python main.py --data_root examples/s3dis
```

We hope you find this 3D point cloud analysis tool useful. If you encounter any problems or have any suggestions for improvements, please open an issue on GitHub. Enjoy analyzing your point clouds!

# License:
Our code is released under MIT License (see LICENSE file for details).
