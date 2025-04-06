# Lift-Plate-Shoot Data Loader Modifications Documentation

This document provides detailed documentation for the modified data loaders in the Lift-Plate-Shoot (LSS) project. The modifications enable support for Fisheye cameras using the KITTI-360 dataset, while retaining the original NuScenes data loader for legacy use.

The documentation covers two main modules:

- **data.py** – The original implementation for the NuScenes dataset.
- **fisheye_data.py** – The modified implementation for the KITTI-360 dataset with Fisheye cameras.

> **Note:** All GitHub URLs are placeholders. Please replace `https://your.repo.url/...` with the actual repository links.

---

## Table of Contents

- [Overview](#overview)
- [data.py Module (NuScenes Data Loader)](#datapy-module)
  - [Class [NuscData](https://your.repo.url/NuscData-placeholder)](#class-nuscdata)
    - [Method [__init__](https://your.repo.url/NuscData.__init__-placeholder)](#method-init)
    - [Method [fix_nuscenes_formatting](https://your.repo.url/NuscData.fix_nuscenes_formatting-placeholder)](#method-fix_nuscenes_formatting)
    - [Method [get_scenes](https://your.repo.url/NuscData.get_scenes-placeholder)](#method-get_scenes)
    - [Method [prepro](https://your.repo.url/NuscData.prepro-placeholder)](#method-prepro)
    - [Method [sample_augmentation](https://your.repo.url/NuscData.sample_augmentation-placeholder)](#method-sample_augmentation)
    - [Method [get_image_data](https://your.repo.url/NuscData.get_image_data-placeholder)](#method-get_image_data)
    - [Method [get_lidar_data](https://your.repo.url/NuscData.get_lidar_data-placeholder)](#method-get_lidar_data)
    - [Method [get_binimg](https://your.repo.url/NuscData.get_binimg-placeholder)](#method-get_binimg)
    - [Method [choose_cams](https://your.repo.url/NuscData.choose_cams-placeholder)](#method-choose_cams)
    - [Method [__str__](https://your.repo.url/NuscData.__str__-placeholder)](#method-str)
    - [Method [__len__](https://your.repo.url/NuscData.__len__-placeholder)](#method-len)
  - [Class [VizData](https://your.repo.url/VizData-placeholder)](#class-vizdata)
    - [Method [__init__](https://your.repo.url/VizData.__init__-placeholder)](#method-vizdata_init)
    - [Method [__getitem__](https://your.repo.url/VizData.__getitem__-placeholder)](#method-vizdata_getitem)
  - [Class [SegmentationData](https://your.repo.url/SegmentationData-placeholder)](#class-segmentationdata)
    - [Method [__init__](https://your.repo.url/SegmentationData.__init__-placeholder)](#method-seg_init)
    - [Method [__getitem__](https://your.repo.url/SegmentationData.__getitem__-placeholder)](#method-seg_getitem)
  - [Function [worker_rnd_init](https://your.repo.url/worker_rnd_init-placeholder)](#function-worker_rnd_init)
  - [Function [compile_data](https://your.repo.url/compile_data-placeholder)](#function-compile_data)
- [fisheye_data.py Module (KITTI-360 Fisheye Data Loader)](#fisheye_datapy-module)
  - [Class [KittiData](https://your.repo.url/KittiData-placeholder)](#class-kittidata)
    - [Method [__init__](https://your.repo.url/KittiData.__init__-placeholder)](#method-kitti_init)
    - [Method [shift_origin](https://your.repo.url/KittiData.shift_origin-placeholder)](#method-shift_origin)
    - [Method [get_sequences](https://your.repo.url/KittiData.get_sequences-placeholder)](#method-get_sequences)
    - [Method [prepro](https://your.repo.url/KittiData.prepro-placeholder)](#method-kitti_prepro)
    - [Method [get_bboxes](https://your.repo.url/KittiData.get_bboxes-placeholder)](#method-get_bboxes)
    - [Method [sample_augmentation](https://your.repo.url/KittiData.sample_augmentation-placeholder)](#method-kitti_sample_augmentation)
    - [Method [get_aug_image_data](https://your.repo.url/KittiData.get_aug_image_data-placeholder)](#method-get_aug_image_data)
    - [Method [get_image_data](https://your.repo.url/KittiData.get_image_data-placeholder)](#method-get_image_data_kitti)
    - [Method [get_lidar_data](https://your.repo.url/KittiData.get_lidar_data-placeholder)](#method-get_lidar_data_kitti)
    - [Method [get_binimg](https://your.repo.url/KittiData.get_binimg-placeholder)](#method-get_binimg_kitti)
    - [Method [get_cams](https://your.repo.url/KittiData.get_cams-placeholder)](#method-get_cams)
    - [Method [__str__](https://your.repo.url/KittiData.__str__-placeholder)](#method-kitti_str)
    - [Method [__len__](https://your.repo.url/KittiData.__len__-placeholder)](#method-kitti_len)
  - [Class [VizData](https://your.repo.url/VizDataKitti-placeholder)](#class-vizdata-kitti)
    - [Method [__init__](https://your.repo.url/VizDataKitti.__init__-placeholder)](#method-viz_kitti_init)
    - [Method [get_colored_binimg](https://your.repo.url/VizDataKitti.get_colored_binimg-placeholder)](#method-get_colored_binimg)
    - [Method [__getitem__](https://your.repo.url/VizDataKitti.__getitem__-placeholder)](#method-viz_kitti_getitem)
  - [Class [SegmentationData](https://your.repo.url/SegmentationDataKitti-placeholder)](#class-segmentationdata-kitti)
    - [Method [__init__](https://your.repo.url/SegmentationDataKitti.__init__-placeholder)](#method-seg_kitti_init)
    - [Method [__getitem__](https://your.repo.url/SegmentationDataKitti.__getitem__-placeholder)](#method-seg_kitti_getitem)
  - [Function [worker_rnd_init](https://your.repo.url/worker_rnd_initKitti-placeholder)](#function-worker_rnd_init_kitti)
  - [Function [compile_data](https://your.repo.url/compile_dataKitti-placeholder)](#function-compile_data_kitti)

---

## Overview

The modifications include:
- **Data Format Adaptation:**  
  The original `data.py` is tailored for the NuScenes dataset. The modified `fisheye_data.py` supports the KITTI-360 format and Fisheye cameras, incorporating necessary changes in file paths, calibration, and annotations.
  
- **Camera & Annotation Handling:**  
  In the KITTI-360 version, new helper modules (e.g., `CameraFisheye`, `Annotation3D`) are used to handle camera models, fisheye distortions, and 3D annotations.

- **Augmentation Pipeline:**  
  Both versions include image augmentation (resize, crop, flip, rotate) but the KITTI-360 loader distinguishes between augmented and non-augmented pipelines (see `get_aug_image_data` vs. `get_image_data`).

- **BEV Generation:**  
  Binary and colored Bird’s-Eye View (BEV) maps are generated using different strategies tailored to each dataset.

---

## data.py Module (NuScenes Data Loader)

### Class [NuscData](https://your.repo.url/NuscData-placeholder)
A PyTorch dataset class that loads and preprocesses data from the NuScenes dataset.

#### Method [__init__](https://your.repo.url/NuscData.__init__-placeholder)
- **Purpose:** Initializes the dataset by:
  - Storing configuration parameters.
  - Loading scenes and samples.
  - Preprocessing samples.
  - Generating the BEV grid using `gen_dx_bx`.
  - Adjusting file paths if necessary.
- **Parameters:**
  - `nusc` (*NuScenes instance*): NuScenes dataset object.
  - `is_train` (*bool*): Indicates training or validation mode.
  - `data_aug_conf` (*dict*): Data augmentation configuration parameters.
  - `grid_conf` (*dict*): Configuration for grid boundaries and resolution.
- **Returns:** An initialized instance of `NuscData`.

#### Method [fix_nuscenes_formatting](https://your.repo.url/NuscData.fix_nuscenes_formatting-placeholder)
- **Purpose:** Ensures that file paths within the NuScenes object match the actual file locations.
- **Parameters:** None.
- **Output:**  
  - *Side-effect:* Updates internal sample records with corrected file paths.
- **Data Type:** None (returns `None`).

#### Method [get_scenes](https://your.repo.url/NuscData.get_scenes-placeholder)
- **Purpose:** Filters and retrieves scene names based on the training or validation split.
- **Parameters:** None.
- **Output:**  
  - *List of strings:* Scene names.
- **Data Type:** `List[str]`

#### Method [prepro](https://your.repo.url/NuscData.prepro-placeholder)
- **Purpose:** Preprocesses the dataset by filtering samples belonging to the chosen scenes and sorting them.
- **Parameters:** None.
- **Output:**  
  - *List:* Processed sample records.
- **Data Type:** `List[dict]`

#### Method [sample_augmentation](https://your.repo.url/NuscData.sample_augmentation-placeholder)
- **Purpose:** Computes augmentation parameters including resize factor, dimensions, crop coordinates, flip flag, and rotation angle.
- **Parameters:** None.
- **Output:**  
  - *Tuple:* `(resize, resize_dims, crop, flip, rotate)`
    - `resize` (*float*)
    - `resize_dims` (*tuple of ints*): New dimensions after resizing.
    - `crop` (*tuple of ints*): Crop coordinates.
    - `flip` (*bool*): Indicates if horizontal flip is applied.
    - `rotate` (*float*): Rotation angle.
- **Data Type:** `Tuple`

#### Method [get_image_data](https://your.repo.url/NuscData.get_image_data-placeholder)
- **Purpose:** Loads images from specified cameras, applies augmentation, and returns image tensors along with camera calibration data.
- **Parameters:**
  - `rec` (*dict*): A sample record from NuScenes.
  - `cams` (*List[str]*): List of camera identifiers.
- **Output:**  
  - *Tuple:* Contains:
    - Images tensor (normalized)
    - Rotation matrices tensor
    - Translation vectors tensor
    - Camera intrinsic matrices tensor
    - Post-augmentation rotation matrices tensor
    - Post-augmentation translation vectors tensor
- **Data Type:** `Tuple[torch.Tensor, ...]`

#### Method [get_lidar_data](https://your.repo.url/NuscData.get_lidar_data-placeholder)
- **Purpose:** Retrieves LiDAR point cloud data for the sample.
- **Parameters:**
  - `rec` (*dict*): A sample record.
  - `nsweeps` (*int*): Number of LiDAR sweeps to aggregate.
- **Output:**  
  - *Tensor:* LiDAR points (first 3 dimensions: x, y, z).
- **Data Type:** `torch.Tensor`

#### Method [get_binimg](https://your.repo.url/NuscData.get_binimg-placeholder)
- **Purpose:** Generates a binary BEV image by projecting object annotations onto the BEV grid.
- **Parameters:**
  - `rec` (*dict*): A sample record.
- **Output:**  
  - *Tensor:* Binary BEV image with a shape compatible with the grid dimensions.
- **Data Type:** `torch.Tensor`

#### Method [choose_cams](https://your.repo.url/NuscData.choose_cams-placeholder)
- **Purpose:** Randomly selects a subset of camera identifiers during training.
- **Parameters:** None.
- **Output:**  
  - *List[str]:* Selected camera identifiers.
- **Data Type:** `List[str]`

#### Method [__str__](https://your.repo.url/NuscData.__str__-placeholder)
- **Purpose:** Returns a string representation summarizing the dataset.
- **Parameters:** None.
- **Output:**  
  - *String:* Summary including the number of samples, split type, and augmentation configuration.
- **Data Type:** `str`

#### Method [__len__](https://your.repo.url/NuscData.__len__-placeholder)
- **Purpose:** Returns the number of samples in the dataset.
- **Parameters:** None.
- **Output:**  
  - *Integer:* Total number of samples.
- **Data Type:** `int`

---

### Class [VizData](https://your.repo.url/VizData-placeholder)
Inherits from `NuscData` for visualization purposes.

#### Method [__init__](https://your.repo.url/VizData.__init__-placeholder)
- **Purpose:** Inherits and initializes all properties from `NuscData`.
- **Parameters:**  
  - Inherits all parameters from `NuscData.__init__`.
- **Output:**  
  - An instance of `VizData`.
- **Data Type:** Instance of `VizData`

#### Method [__getitem__](https://your.repo.url/VizData.__getitem__-placeholder)
- **Purpose:** Retrieves a complete sample for visualization including image data, LiDAR data, and the binary BEV image.
- **Parameters:**
  - `index` (*int*): Index of the sample.
- **Output:**  
  - *Tuple:* `(imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg)`
    - `imgs` (*torch.Tensor*): Augmented images.
    - `rots` (*torch.Tensor*): Rotation matrices.
    - `trans` (*torch.Tensor*): Translation vectors.
    - `intrins` (*torch.Tensor*): Camera intrinsics.
    - `post_rots` (*torch.Tensor*): Post-augmentation rotations.
    - `post_trans` (*torch.Tensor*): Post-augmentation translations.
    - `lidar_data` (*torch.Tensor*): LiDAR point cloud.
    - `binimg` (*torch.Tensor*): Binary BEV image.
- **Data Type:** `Tuple`

---

### Class [SegmentationData](https://your.repo.url/SegmentationData-placeholder)
Specialized for segmentation tasks; inherits from `NuscData`.

#### Method [__init__](https://your.repo.url/SegmentationData.__init__-placeholder)
- **Purpose:** Inherits initialization from `NuscData`.
- **Parameters:**  
  - Inherits all parameters from `NuscData.__init__`.
- **Output:**  
  - An instance of `SegmentationData`.
- **Data Type:** Instance of `SegmentationData`

#### Method [__getitem__](https://your.repo.url/SegmentationData.__getitem__-placeholder)
- **Purpose:** Retrieves a sample for segmentation tasks including image and BEV data.
- **Parameters:**
  - `index` (*int*): Index of the sample.
- **Output:**  
  - *Tuple:* `(imgs, rots, trans, intrins, post_rots, post_trans, binimg)`
- **Data Type:** `Tuple`

---

### Function [worker_rnd_init](https://your.repo.url/worker_rnd_init-placeholder)
- **Purpose:** Initializes a random seed for data loader workers to ensure reproducibility.
- **Parameters:**
  - `x` (*int*): Worker index.
- **Output:**  
  - None (side-effect: sets the NumPy random seed).
- **Data Type:** `None`

---

### Function [compile_data](https://your.repo.url/compile_data-placeholder)
- **Purpose:** Compiles training and validation DataLoaders.
- **Parameters:**
  - `version` (*str*): Dataset version (e.g., `'trainval'`, `'mini'`).
  - `dataroot` (*str*): Path to the dataset root.
  - `data_aug_conf` (*dict*): Data augmentation configuration.
  - `grid_conf` (*dict*): Grid configuration for BEV.
  - `bsz` (*int*): Batch size.
  - `nworkers` (*int*): Number of workers.
  - `parser_name` (*str*): Specifies which parser to use (`'vizdata'` or `'segmentationdata'`).
- **Output:**  
  - *Tuple:* `(trainloader, valloader)` – PyTorch DataLoader instances.
- **Data Type:** `Tuple[DataLoader, DataLoader]`

---

## fisheye_data.py Module (KITTI-360 Fisheye Data Loader)

### Class [KittiData](https://your.repo.url/KittiData-placeholder)
A PyTorch dataset class designed for the KITTI-360 dataset with fisheye camera support.

#### Method [__init__](https://your.repo.url/KittiData.__init__-placeholder)
- **Purpose:** Initializes the dataset by:
  - Setting environment paths (expects `KITTI360_DATASET` to be set).
  - Loading sequences, poses, and annotations.
  - Computing the BEV grid and additional transformation matrices.
- **Parameters:**
  - `is_train` (*bool*): Training mode flag.
  - `data_aug_conf` (*dict*): Data augmentation parameters.
  - `grid_conf` (*dict*): Grid configuration (boundaries and resolution).
  - `is_aug` (*bool*, optional): Whether to use additional augmentation (default is `False`).
- **Output:**  
  - An instance of `KittiData`.
- **Data Type:** Instance of `KittiData`

#### Method [shift_origin](https://your.repo.url/KittiData.shift_origin-placeholder)
- **Purpose:** Applies an additional shift transformation to the BEV coordinate system.
- **Parameters:**
  - `x` (*float*, default `-0.81`): Shift along the X-axis.
  - `y` (*float*, default `-0.32`): Shift along the Y-axis.
  - `z` (*float*, default `-0.9`): Shift along the Z-axis.
- **Output:**  
  - *NumPy array:* 4×4 transformation matrix.
- **Data Type:** `np.ndarray`

#### Method [get_sequences](https://your.repo.url/KittiData.get_sequences-placeholder)
- **Purpose:** Retrieves and splits sequence names into training and validation sets.
- **Parameters:**
  - `val_idxs` (*List[int]*, optional): Indices to designate validation sequences (default `[8]`).
- **Output:**  
  - *List:* Sequence names for the current mode (training or validation).
- **Data Type:** `List[str]`

#### Method [prepro](https://your.repo.url/KittiData.prepro-placeholder)
- **Purpose:** Processes sequences by:
  - Loading poses from text files.
  - Aligning frames with poses.
  - Packaging the data into a structured NumPy array.
- **Parameters:** None.
- **Output:**  
  - *NumPy structured array:* Contains `sequence` (str), `frame` (str), and `pose` (4×4 flattened, float32).
- **Data Type:** `np.ndarray`

#### Method [get_bboxes](https://your.repo.url/KittiData.get_bboxes-placeholder)
- **Purpose:** Loads 3D bounding box annotations for each sequence.
- **Parameters:**
  - `sequences` (*List[str]*): List of sequence names.
- **Output:**  
  - *Dictionary:* Mapping sequence names to their bounding box objects.
- **Data Type:** `Dict[str, Any]`

#### Method [sample_augmentation](https://your.repo.url/KittiData.sample_augmentation-placeholder)
- **Purpose:** Computes image augmentation parameters (resize, crop, flip, rotation) similar to `NuscData` but tailored for KITTI-360.
- **Parameters:** None.
- **Output:**  
  - *Tuple:* `(resize, resize_dims, crop, flip, rotate)`
- **Data Type:** `Tuple`

#### Method [get_aug_image_data](https://your.repo.url/KittiData.get_aug_image_data-placeholder)
- **Purpose:** Retrieves and augments image data using fisheye camera calibration parameters.
- **Parameters:**
  - `rec` (*dict*): A sample record.
  - `cams` (*dict*): Dictionary of camera objects.
- **Output:**  
  - *Tuple:* Contains augmented images, rotation matrices, translation vectors, intrinsic parameters, and post-augmentation matrices.
- **Data Type:** `Tuple[torch.Tensor, ...]`

#### Method [get_image_data](https://your.repo.url/KittiData.get_image_data-placeholder)
- **Purpose:** Loads image data from the KITTI-360 dataset, handling fisheye distortions and camera projection.
- **Parameters:**
  - `rec` (*dict*): A sample record.
  - `cams` (*dict*): Dictionary mapping camera identifiers to fisheye camera objects.
- **Output:**  
  - *Tuple:* `(imgs, rots, trans, intrinsics, distortions, xis)`
    - `imgs` (*torch.Tensor*): Normalized image tensors.
    - `rots` (*torch.Tensor*): Rotation matrices.
    - `trans` (*torch.Tensor*): Translation vectors.
    - `intrinsics` (*torch.Tensor*): Intrinsic parameter vectors.
    - `distortions` (*torch.Tensor*): Distortion coefficients.
    - `xis` (*torch.Tensor*): Fisheye-specific xi parameter.
- **Data Type:** `Tuple[torch.Tensor, ...]`

#### Method [get_lidar_data](https://your.repo.url/KittiData.get_lidar_data-placeholder)
- **Purpose:** Retrieves LiDAR data for a given sample.
- **Parameters:**
  - `rec` (*dict*): A sample record.
  - `nsweeps` (*int*): Number of LiDAR sweeps.
- **Output:**  
  - *Tensor:* LiDAR point cloud (first 3 dimensions: x, y, z).
- **Data Type:** `torch.Tensor`

#### Method [get_binimg](https://your.repo.url/KittiData.get_binimg-placeholder)
- **Purpose:** Generates a binary BEV image by transforming 3D bounding boxes to the BEV plane.
- **Parameters:**
  - `rec` (*dict*): A sample record.
- **Output:**  
  - *Tensor:* Binary BEV image.
- **Data Type:** `torch.Tensor`

#### Method [get_cams](https://your.repo.url/KittiData.get_cams-placeholder)
- **Purpose:** Selects the camera objects based on the configuration and whether augmentation is applied.
- **Parameters:** None.
- **Output:**  
  - *Dictionary:* Mapping camera identifiers to fisheye camera objects.
- **Data Type:** `Dict[str, Any]`

#### Method [__str__](https://your.repo.url/KittiData.__str__-placeholder)
- **Purpose:** Returns a summary string representation of the dataset.
- **Parameters:** None.
- **Output:**  
  - *String:* Summary including sample count, mode, and augmentation configuration.
- **Data Type:** `str`

#### Method [__len__](https://your.repo.url/KittiData.__len__-placeholder)
- **Purpose:** Returns the total number of samples.
- **Parameters:** None.
- **Output:**  
  - *Integer:* Total sample count.
- **Data Type:** `int`

---

### Class [VizData (KITTI-360)](https://your.repo.url/VizDataKitti-placeholder)
Specialized for visualization, extends `KittiData` and provides additional BEV map functionalities.

#### Method [__init__](https://your.repo.url/VizDataKitti.__init__-placeholder)
- **Purpose:** Initializes the visualization dataset, inheriting properties from `KittiData`.
- **Parameters:**  
  - Inherits all parameters from `KittiData.__init__`.
- **Output:**  
  - An instance of `VizData` for KITTI-360.
- **Data Type:** Instance of `VizData`

#### Method [get_colored_binimg](https://your.repo.url/VizDataKitti.get_colored_binimg-placeholder)
- **Purpose:** Generates a colored BEV map by overlaying fisheye coverage, ego-vehicle visualization, and annotated objects.
- **Parameters:**
  - `rec` (*dict*): A sample record.
  - `cams` (*dict*): Dictionary of camera objects.
- **Output:**  
  - *Tensor:* Colored BEV image.
- **Data Type:** `torch.Tensor`

#### Method [__getitem__](https://your.repo.url/VizDataKitti.__getitem__-placeholder)
- **Purpose:** Retrieves a complete sample for visualization including:
  - Image data.
  - Camera extrinsic/intrinsic parameters.
  - Binary and colored BEV maps.
- **Parameters:**
  - `index` (*int*): Index of the sample.
- **Output:**  
  - *Tuple:* `(imgs, rots, trans, K, D, xi, binimg, colored_binimg)`
- **Data Type:** `Tuple[torch.Tensor, ...]`

---

### Class [SegmentationData (KITTI-360)](https://your.repo.url/SegmentationDataKitti-placeholder)
Specialized for segmentation tasks on KITTI-360 data, extends `KittiData`.

#### Method [__init__](https://your.repo.url/SegmentationDataKitti.__init__-placeholder)
- **Purpose:** Inherits initialization from `KittiData`.
- **Parameters:**  
  - Inherits all parameters from `KittiData.__init__`.
- **Output:**  
  - An instance of `SegmentationData`.
- **Data Type:** Instance of `SegmentationData`

#### Method [__getitem__](https://your.repo.url/SegmentationDataKitti.__getitem__-placeholder)
- **Purpose:** Retrieves a segmentation sample including image data and the binary BEV map.
- **Parameters:**
  - `index` (*int*): Index of the sample.
- **Output:**  
  - *Tuple:* `(imgs, rots, trans, K, D, xi, binimg)`
- **Data Type:** `Tuple[torch.Tensor, ...]`

---

### Function [worker_rnd_init (KITTI-360)](https://your.repo.url/worker_rnd_initKitti-placeholder)
- **Purpose:** Initializes the random seed for KITTI-360 data loader workers.
- **Parameters:**
  - `x` (*int*): Worker index.
- **Output:**  
  - None (side-effect: sets NumPy random seed).
- **Data Type:** `None`

---

### Function [compile_data (KITTI-360)](https://your.repo.url/compile_dataKitti-placeholder)
- **Purpose:** Compiles training and validation DataLoaders for the KITTI-360 fisheye data.
- **Parameters:**
  - `data_aug_conf` (*dict*): Data augmentation configuration.
  - `grid_conf` (*dict*): BEV grid configuration.
  - `is_aug` (*bool*): Flag indicating whether additional augmentation is used.
  - `bsz` (*int*): Batch size.
  - `nworkers` (*int*): Number of workers.
  - `parser_name` (*str*): Specifies parser type (`'vizdata'` or `'segmentationdata'`).
- **Output:**  
  - *Tuple:* `(trainloader, valloader)` – PyTorch DataLoader instances.
- **Data Type:** `Tuple[DataLoader, DataLoader]`

---

## Final Notes

- **Placeholder Text:**  
  Some descriptions are marked as "props" where specific details are not fully known. Please update these sections with exact descriptions based on your project requirements.

- **GitHub URLs:**  
  Replace all placeholder URLs (`https://your.repo.url/...`) with the correct links to the corresponding code locations in your repository.

This comprehensive documentation is intended as a handover guide to facilitate the understanding and further development of the modified data loaders for the LSS project.
