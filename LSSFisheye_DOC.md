# Lift-Splat-Shoot Data Loader Documentation

This documentation describes the modified data loader components for the Lift-Splat-Shoot (LSS) model to support Fisheye cameras with the KITTI-360 dataset. The modifications build on the original NuScenes-compatible data loader and include significant changes in data formatting, camera calibration, BEV generation, and data augmentation.

**Repository URL Placeholder:**  
[GitHub Repository](https://github.com/YourRepo/YourProject)  
*(Replace with your actual GitHub repository URL and update specific links as needed.)*

---

## Overview

The project comprises two main implementations:

1. **Original LSS Data Loader (`data.py`):**  
   This version is designed for the NuScenes dataset. It provides functionalities to load image and lidar data, perform data augmentation, generate BEV maps, and handle dataset splits based on NuScenes' structure.

2. **Modified Fisheye Data Loader (`fisheye_data.py`):**  
   Adapted for the KITTI-360 dataset, this loader supports Fisheye camera models. The modifications include new environment-based path setup, a custom fisheye calibration interface, adjusted sample preprocessing, and BEV generation tailored for KITTI-360.

---

## Table of Contents

- [Data Loader for NuScenes (`data.py`)](#nuscdata)
  - [Classes](#nuscdata-classes)
    - [NuscData](#nuscdata)
      - [Methods](#nuscdata-methods)
    - [VizData](#vizdata-datapy)
      - [Methods](#vizdata-datapy-methods)
    - [SegmentationData](#segmentationdata-datapy)
      - [Methods](#segmentationdata-datapy-methods)
  - [Functions](#nuscdata-functions)
    - [worker_rnd_init](#worker_rnd_init)
    - [compile_data](#compile_data)
- [Data Loader for KITTI-360 (`fisheye_data.py`)](#kittidata)
  - [Classes](#kittidata-classes)
    - [KittiData](#kittidata)
      - [Methods](#kittidata-methods)
    - [VizData](#vizdata-fisheye)
      - [Methods](#vizdata-fisheye-methods)
    - [SegmentationData](#segmentationdata-fisheye)
      - [Methods](#segmentationdata-fisheye-methods)
  - [Functions](#kittidata-functions)
    - [worker_rnd_init (fisheye)](#worker_rnd_init-fisheye)
    - [compile_data (fisheye)](#compile_data-fisheye)

---

## Data Loader for NuScenes (`data.py`)

### Classes

#### NuscData  
*Location on GitHub: [NuscData](https://github.com/YourRepo/YourProject/path/to/data.py#LXX)*

**Purpose:**  
Provides a dataset loader for the NuScenes dataset. It handles loading of images, lidar data, and BEV map generation, along with applying data augmentation and correcting file paths.

**Key Methods:**

- **`__init__(self, nusc, is_train, data_aug_conf, grid_conf)`**  
  **Description:**  
  Initializes the dataset with a NuScenes object, training flag, augmentation configuration, and grid configuration. It pre-processes samples, generates grid bounds, and adjusts file paths if necessary.
  
  **Arguments:**  
  - `nusc`: NuScenes object containing dataset information.  
  - `is_train`: Boolean indicating if the dataset is used for training.  
  - `data_aug_conf`: Dictionary with data augmentation parameters.  
  - `grid_conf`: Dictionary with grid configuration for BEV mapping.
  
- **`fix_nuscenes_formatting(self)`**  
  **Description:**  
  Adjusts file paths within the NuScenes object if the default file paths do not exist.
  
- **`get_scenes(self)`**  
  **Description:**  
  Retrieves the scene split (train or val) based on the dataset version.
  
- **`prepro(self)`**  
  **Description:**  
  Preprocesses the sample data, filters by scene, and sorts the samples.
  
- **`sample_augmentation(self)`**  
  **Description:**  
  Determines augmentation parameters such as resizing, cropping, flipping, and rotation.
  
- **`get_image_data(self, rec, cams)`**  
  **Description:**  
  Loads and processes image data for the specified camera channels, applying augmentation and returning transformed images, intrinsic parameters, rotation, and translation matrices.
  
- **`get_lidar_data(self, rec, nsweeps)`**  
  **Description:**  
  Retrieves lidar data from a given sample record and returns the relevant points.
  
- **`get_binimg(self, rec)`**  
  **Description:**  
  Generates a binary BEV image by projecting object bounding boxes onto a grid.
  
- **`choose_cams(self)`**  
  **Description:**  
  Selects camera channels based on training conditions and configuration.
  
- **`__len__(self)`**  
  **Description:**  
  Returns the number of samples in the dataset.

*(For complete method details, refer to the source code at [data.py](https://github.com/YourRepo/YourProject/path/to/data.py#LXX).)*  
:contentReference[oaicite:0]{index=0}

---

#### VizData (data.py)  
*Location on GitHub: [VizData](https://github.com/YourRepo/YourProject/path/to/data.py#LYY)*

**Purpose:**  
A subclass of `NuscData` specifically designed for visualization. It returns additional lidar and BEV binary images alongside the image data.

**Key Methods:**

- **`__getitem__(self, index)`**  
  **Description:**  
  Retrieves and returns the processed image data, lidar data, and BEV binary image for visualization.
  
*(For more details, refer to the source code at [VizData](https://github.com/YourRepo/YourProject/path/to/data.py#LYY).)*  
:contentReference[oaicite:1]{index=1}

---

#### SegmentationData (data.py)  
*Location on GitHub: [SegmentationData](https://github.com/YourRepo/YourProject/path/to/data.py#LZZ)*

**Purpose:**  
A subclass of `NuscData` tailored for segmentation tasks. It returns the image data along with the BEV binary image without including lidar data.

**Key Methods:**

- **`__getitem__(self, index)`**  
  **Description:**  
  Returns the image data and BEV binary image for segmentation purposes.
  
*(For complete details, see [SegmentationData](https://github.com/YourRepo/YourProject/path/to/data.py#LZZ).)*  
:contentReference[oaicite:2]{index=2}

---

### Functions

#### worker_rnd_init  
*Location on GitHub: [worker_rnd_init](https://github.com/YourRepo/YourProject/path/to/data.py#LAA)*

**Purpose:**  
Initializes the random seed for each worker to ensure reproducibility.

**Arguments:**  
- `x`: Worker index used to seed the random number generator.

---

#### compile_data  
*Location on GitHub: [compile_data](https://github.com/YourRepo/YourProject/path/to/data.py#LBB)*

**Purpose:**  
Creates DataLoader instances for training and validation using either `VizData` or `SegmentationData`.

**Arguments:**  
- `version`: NuScenes dataset version string.  
- `dataroot`: Root path for the dataset.  
- `data_aug_conf`: Dictionary containing data augmentation configurations.  
- `grid_conf`: Dictionary containing grid configuration parameters.  
- `bsz`: Batch size for the DataLoader.  
- `nworkers`: Number of worker processes for data loading.  
- `parser_name`: A string indicating which data parser to use (`vizdata` or `segmentationdata`).

*(Further details available at [compile_data](https://github.com/YourRepo/YourProject/path/to/data.py#LBB).)*  
:contentReference[oaicite:3]{index=3}

---

## Data Loader for KITTI-360 (`fisheye_data.py`)

### Classes

#### KittiData  
*Location on GitHub: [KittiData](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LC1)*

**Purpose:**  
Serves as the base class for the KITTI-360 dataset loader. It handles environment setup, sequence preprocessing, camera calibration for fisheye cameras, and BEV generation specific to the KITTI-360 dataset.

**Key Methods:**

- **`__init__(self, is_train, data_aug_conf, grid_conf, is_aug=False)`**  
  **Description:**  
  Initializes paths based on environment variables, sets up sequences, loads bounding boxes, and configures the BEV grid. Also defines camera selection and additional coordinate shifting.
  
  **Arguments:**  
  - `is_train`: Boolean indicating training or validation mode.  
  - `data_aug_conf`: Dictionary with augmentation parameters.  
  - `grid_conf`: Dictionary with grid configuration.  
  - `is_aug`: Optional boolean for augmented data handling.
  
- **`shift_origin(self, x=-0.81, y=-0.32, z=-0.9)`**  
  **Description:**  
  Applies an additional translation to adjust the coordinate system for BEV mapping.
  
- **`get_sequences(self, val_idxs=[8])`**  
  **Description:**  
  Retrieves and splits sequence names from the dataset directory into training and validation sets.
  
- **`prepro(self)`**  
  **Description:**  
  Preprocesses each sequence by loading pose data, constructing a structured NumPy array, and handling sequence-specific issues.
  
- **`get_bboxes(self, sequences)`**  
  **Description:**  
  Loads 3D bounding boxes for each sequence using a custom annotation loader.
  
- **`sample_augmentation(self)`**  
  **Description:**  
  Defines augmentation parameters (resize, crop, flip, rotation) similar to the original loader but adjusted for KITTI-360.
  
- **`get_image_data(self, rec, cams)`**  
  **Description:**  
  Loads and processes fisheye image data, extracting camera intrinsic parameters (including distortion and fisheye-specific parameters), extrinsic parameters, and performing normalization.
  
- **`get_aug_image_data(self, rec, cams)`**  
  **Description:**  
  (Experimental) Applies augmentation to fisheye images and returns augmented image data.
  
- **`get_lidar_data(self, rec, nsweeps)`**  
  **Description:**  
  Loads lidar point cloud data similarly to the NuScenes version.
  
- **`get_binimg(self, rec)`**  
  **Description:**  
  Generates a binary BEV image by projecting 3D bounding boxes onto a BEV grid using KITTI-360’s annotation and pose data.
  
- **`get_cams(self)`**  
  **Description:**  
  Selects and configures camera data using the fisheye calibration model.
  
- **`__len__(self)`**  
  **Description:**  
  Returns the number of samples in the dataset.

*(For full implementation details, refer to [KittiData](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LC1).)*  
:contentReference[oaicite:4]{index=4}

---

#### VizData (fisheye_data.py)  
*Location on GitHub: [VizData (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LD1)*

**Purpose:**  
A subclass of `KittiData` for visualization purposes. It provides both binary and colored BEV map generation, along with standard image data for fisheye cameras.

**Key Methods:**

- **`__getitem__(self, index)`**  
  **Description:**  
  Retrieves and returns processed fisheye image data, BEV maps (binary and colored), and other related parameters for visualization.
  
- **`get_colored_binimg(self, rec, cams)`**  
  **Description:**  
  Generates a colored BEV image by drawing fisheye camera coverage and annotated bounding boxes in a color-coded manner.
  
*(For complete method details, see [VizData (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LD1).)*  
:contentReference[oaicite:5]{index=5}

---

#### SegmentationData (fisheye_data.py)  
*Location on GitHub: [SegmentationData (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LE1)*

**Purpose:**  
A subclass of `KittiData` optimized for segmentation tasks. It returns fisheye image data and the corresponding binary BEV map.

**Key Methods:**

- **`__getitem__(self, index)`**  
  **Description:**  
  Provides the image data and binary BEV map without the additional visualization elements used in the VizData class.
  
*(For full details, see [SegmentationData (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LE1).)*  
:contentReference[oaicite:6]{index=6}

---

### Functions

#### worker_rnd_init (fisheye_data.py)  
*Location on GitHub: [worker_rnd_init (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LF1)*

**Purpose:**  
Sets the random seed for each worker process to ensure reproducibility during data loading.

**Arguments:**  
- `x`: Worker index used to seed the random number generator.

---

#### compile_data (fisheye_data.py)  
*Location on GitHub: [compile_data (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LG1)*

**Purpose:**  
Constructs DataLoader instances for training and validation from the KITTI-360 dataset, using either the VizData or SegmentationData parser.

**Arguments:**  
- `data_aug_conf`: Dictionary with data augmentation parameters.  
- `grid_conf`: Dictionary with BEV grid configuration.  
- `is_aug`: Boolean flag for applying additional augmentation.  
- `bsz`: Batch size for data loading.  
- `nworkers`: Number of worker processes for loading data.  
- `parser_name`: String specifying the data parser type (`vizdata` or `segmentationdata`).

*(For further details, check [compile_data (Fisheye)](https://github.com/YourRepo/YourProject/path/to/fisheye_data.py#LG1).)*  
:contentReference[oaicite:7]{index=7}

---

## Final Notes

- **Calibration and Augmentation:**  
  Special attention should be given to the fisheye calibration parameters and augmentation configurations to ensure compatibility with the KITTI-360 sensor suite.

- **BEV Mapping:**  
  The modifications in BEV generation, including coordinate shifting and polygon filling, are crucial for correct spatial representation in the KITTI-360 context.

- **Further Improvements:**  
  Future documentation updates can include more detailed descriptions and examples once additional insights or refinements are made.

---

*This documentation is intended to serve as a handover guide. Please update the GitHub URL placeholders with the corresponding links from your repository once available.*
