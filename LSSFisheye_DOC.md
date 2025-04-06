# Lift-Plate-Shoot Data Loader Modifications Documentation

This document provides detailed documentation for the modified data loaders in the Lift-Plate-Shoot (LSS) project. The modifications enable support for Fisheye cameras using the KITTI-360 dataset, while retaining the original NuScenes data loader for legacy use.

The documentation covers two main modules:

- **data.py** – The original implementation for the NuScenes dataset.
- **fisheye_data.py** – The modified implementation for the KITTI-360 dataset with Fisheye cameras.

> **Note:** All GitHub URLs are placeholders. Please replace `https://your.repo.url/...` with the actual repository links.

---

## Table of Contents

- [Overview](#overview)
- [data.py Module (NuScenes Data Loader)](#datapy-module-nuscenes-data-loader)
  - [NuscData](#nuscdata)
    - [__init__](#nuscdata-init)
    - [fix_nuscenes_formatting](#nuscdata-fix_nuscenes_formatting)
    - [get_scenes](#nuscdata-get_scenes)
    - [prepro](#nuscdata-prepro)
    - [sample_augmentation](#nuscdata-sample_augmentation)
    - [get_image_data](#nuscdata-get_image_data)
    - [get_lidar_data](#nuscdata-get_lidar_data)
    - [get_binimg](#nuscdata-get_binimg)
    - [choose_cams](#nuscdata-choose_cams)
    - [__str__](#nuscdata-str)
    - [__len__](#nuscdata-len)
  - [VizData](#vizdata)
    - [__init__](#vizdata-init)
    - [__getitem__](#vizdata-getitem)
  - [SegmentationData](#segmentationdata)
    - [__init__](#segmentationdata-init)
    - [__getitem__](#segmentationdata-getitem)
  - [worker_rnd_init](#worker_rnd_init)
  - [compile_data](#compile_data)
- [fisheye_data.py Module (KITTI-360 Fisheye Data Loader)](#fisheye_datapy-module-kitti-360-fisheye-data-loader)
  - [KittiData](#kittidata)
    - [__init__](#kittidata-init)
    - [shift_origin](#kittidata-shift_origin)
    - [get_sequences](#kittidata-get_sequences)
    - [prepro](#kittidata-prepro)
    - [get_bboxes](#kittidata-get_bboxes)
    - [sample_augmentation](#kittidata-sample_augmentation)
    - [get_aug_image_data](#kittidata-get_aug_image_data)
    - [get_image_data](#kittidata-get_image_data)
    - [get_lidar_data](#kittidata-get_lidar_data)
    - [get_binimg](#kittidata-get_binimg)
    - [get_cams](#kittidata-get_cams)
    - [__str__](#kittidata-str)
    - [__len__](#kittidata-len)
  - [VizData (KITTI-360)](#vizdata-kitti-360)
    - [__init__](#vizdata-kitti-360-init)
    - [get_colored_binimg](#vizdata-kitti-360-get_colored_binimg)
    - [__getitem__](#vizdata-kitti-360-getitem)
  - [SegmentationData (KITTI-360)](#segmentationdata-kitti-360)
    - [__init__](#segmentationdata-kitti-360-init)
    - [__getitem__](#segmentationdata-kitti-360-getitem)
  - [worker_rnd_init (KITTI-360)](#worker_rnd_init-kitti-360)
  - [compile_data (KITTI-360)](#compile_data-kitti-360)

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

### NuscData
A PyTorch dataset class that loads and preprocesses data from the NuScenes dataset.  
**GitHub:** [NuscData](https://your.repo.url/NuscData-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/NuscData.__init__-placeholder)  
**Purpose:**  
Initializes the dataset by:
- Storing configuration parameters.
- Loading scenes and samples.
- Preprocessing samples.
- Generating the BEV grid using `gen_dx_bx`.
- Adjusting file paths if necessary.

**Parameters:**
- `nusc` (*NuScenes instance*): NuScenes dataset object.
- `is_train` (*bool*): Indicates training or validation mode.
- `data_aug_conf` (*dict*): Data augmentation configuration parameters.
- `grid_conf` (*dict*): Configuration for grid boundaries and resolution.

**Returns:**  
An initialized instance of `NuscData`.

#### fix_nuscenes_formatting
**GitHub:** [fix_nuscenes_formatting](https://your.repo.url/NuscData.fix_nuscenes_formatting-placeholder)  
**Purpose:**  
Ensures that file paths within the NuScenes object match the actual file locations.

**Parameters:**  
None.

**Output:**  
- *Side-effect:* Updates internal sample records with corrected file paths.  
- *Returns:* `None`

#### get_scenes
**GitHub:** [get_scenes](https://your.repo.url/NuscData.get_scenes-placeholder)  
**Purpose:**  
Filters and retrieves scene names based on the training or validation split.

**Parameters:**  
None.

**Output:**  
- *List of strings:* Scene names.  
- *Data Type:* `List[str]`

#### prepro
**GitHub:** [prepro](https://your.repo.url/NuscData.prepro-placeholder)  
**Purpose:**  
Preprocesses the dataset by filtering samples belonging to the chosen scenes and sorting them.

**Parameters:**  
None.

**Output:**  
- *List:* Processed sample records.  
- *Data Type:* `List[dict]`

#### sample_augmentation
**GitHub:** [sample_augmentation](https://your.repo.url/NuscData.sample_augmentation-placeholder)  
**Purpose:**  
Computes augmentation parameters including resize factor, dimensions, crop coordinates, flip flag, and rotation angle.

**Parameters:**  
None.

**Output:**  
- *Tuple:* `(resize, resize_dims, crop, flip, rotate)` where:
  - `resize` (*float*)
  - `resize_dims` (*tuple of ints*): New dimensions after resizing.
  - `crop` (*tuple of ints*): Crop coordinates.
  - `flip` (*bool*): Indicates if horizontal flip is applied.
  - `rotate` (*float*): Rotation angle.

**Data Type:** `Tuple`

#### get_image_data
**GitHub:** [get_image_data](https://your.repo.url/NuscData.get_image_data-placeholder)  
**Purpose:**  
Loads images from specified cameras, applies augmentation, and returns image tensors along with camera calibration data.

**Parameters:**
- `rec` (*dict*): A sample record from NuScenes.
- `cams` (*List[str]*): List of camera identifiers.

**Output:**  
- *Tuple:* Contains:
  - `imgs` (*torch.Tensor*): Normalized image tensors.
  - `rots` (*torch.Tensor*): Rotation matrices.
  - `trans` (*torch.Tensor*): Translation vectors.
  - `intrins` (*torch.Tensor*): Camera intrinsic matrices.
  - `post_rots` (*torch.Tensor*): Post-augmentation rotation matrices.
  - `post_trans` (*torch.Tensor*): Post-augmentation translation vectors.

**Data Type:** `Tuple[torch.Tensor, ...]`

#### get_lidar_data
**GitHub:** [get_lidar_data](https://your.repo.url/NuscData.get_lidar_data-placeholder)  
**Purpose:**  
Retrieves LiDAR point cloud data for the sample.

**Parameters:**
- `rec` (*dict*): A sample record.
- `nsweeps` (*int*): Number of LiDAR sweeps to aggregate.

**Output:**  
- *Tensor:* LiDAR points (first 3 dimensions: x, y, z).

**Data Type:** `torch.Tensor`

#### get_binimg
**GitHub:** [get_binimg](https://your.repo.url/NuscData.get_binimg-placeholder)  
**Purpose:**  
Generates a binary BEV image by projecting object annotations onto the BEV grid.

**Parameters:**
- `rec` (*dict*): A sample record.

**Output:**  
- *Tensor:* Binary BEV image with a shape compatible with the grid dimensions.

**Data Type:** `torch.Tensor`

#### choose_cams
**GitHub:** [choose_cams](https://your.repo.url/NuscData.choose_cams-placeholder)  
**Purpose:**  
Randomly selects a subset of camera identifiers during training.

**Parameters:**  
None.

**Output:**  
- *List[str]:* Selected camera identifiers.

**Data Type:** `List[str]`

#### __str__
**GitHub:** [__str__](https://your.repo.url/NuscData.__str__-placeholder)  
**Purpose:**  
Returns a string representation summarizing the dataset.

**Parameters:**  
None.

**Output:**  
- *String:* Summary including the number of samples, split type, and augmentation configuration.

**Data Type:** `str`

#### __len__
**GitHub:** [__len__](https://your.repo.url/NuscData.__len__-placeholder)  
**Purpose:**  
Returns the number of samples in the dataset.

**Parameters:**  
None.

**Output:**  
- *Integer:* Total number of samples.

**Data Type:** `int`

---

### VizData
Inherits from `NuscData` for visualization purposes.  
**GitHub:** [VizData](https://your.repo.url/VizData-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/VizData.__init__-placeholder)  
**Purpose:**  
Inherits and initializes all properties from `NuscData`.

**Parameters:**  
- Inherits all parameters from `NuscData.__init__`.

**Output:**  
- An instance of `VizData`.

**Data Type:** Instance of `VizData`

#### __getitem__
**GitHub:** [__getitem__](https://your.repo.url/VizData.__getitem__-placeholder)  
**Purpose:**  
Retrieves a complete sample for visualization including image data, LiDAR data, and the binary BEV image.

**Parameters:**
- `index` (*int*): Index of the sample.

**Output:**  
- *Tuple:* `(imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg)` where:
  - `imgs` (*torch.Tensor*): Augmented images.
  - `rots` (*torch.Tensor*): Rotation matrices.
  - `trans` (*torch.Tensor*): Translation vectors.
  - `intrins` (*torch.Tensor*): Camera intrinsics.
  - `post_rots` (*torch.Tensor*): Post-augmentation rotations.
  - `post_trans` (*torch.Tensor*): Post-augmentation translations.
  - `lidar_data` (*torch.Tensor*): LiDAR point cloud.
  - `binimg` (*torch.Tensor*): Binary BEV image.

**Data Type:** `Tuple[torch.Tensor, ...]`

---

### SegmentationData
Specialized for segmentation tasks; inherits from `NuscData`.  
**GitHub:** [SegmentationData](https://your.repo.url/SegmentationData-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/SegmentationData.__init__-placeholder)  
**Purpose:**  
Inherits initialization from `NuscData`.

**Parameters:**  
- Inherits all parameters from `NuscData.__init__`.

**Output:**  
- An instance of `SegmentationData`.

**Data Type:** Instance of `SegmentationData`

#### __getitem__
**GitHub:** [__getitem__](https://your.repo.url/SegmentationData.__getitem__-placeholder)  
**Purpose:**  
Retrieves a sample for segmentation tasks including image data and the BEV binary image.

**Parameters:**
- `index` (*int*): Index of the sample.

**Output:**  
- *Tuple:* `(imgs, rots, trans, intrins, post_rots, post_trans, binimg)`

**Data Type:** `Tuple[torch.Tensor, ...]`

---

### worker_rnd_init
**GitHub:** [worker_rnd_init](https://your.repo.url/worker_rnd_init-placeholder)  
**Purpose:**  
Initializes a random seed for data loader workers to ensure reproducibility.

**Parameters:**
- `x` (*int*): Worker index.

**Output:**  
- *Returns:* `None` (side-effect: sets the NumPy random seed).

**Data Type:** `None`

---

### compile_data
**GitHub:** [compile_data](https://your.repo.url/compile_data-placeholder)  
**Purpose:**  
Compiles training and validation DataLoaders.

**Parameters:**
- `version` (*str*): Dataset version (e.g., `'trainval'`, `'mini'`).
- `dataroot` (*str*): Path to the dataset root.
- `data_aug_conf` (*dict*): Data augmentation configuration.
- `grid_conf` (*dict*): Grid configuration for BEV.
- `bsz` (*int*): Batch size.
- `nworkers` (*int*): Number of workers.
- `parser_name` (*str*): Specifies which parser to use (`'vizdata'` or `'segmentationdata'`).

**Output:**  
- *Tuple:* `(trainloader, valloader)` – PyTorch DataLoader instances.

**Data Type:** `Tuple[DataLoader, DataLoader]`

---

## fisheye_data.py Module (KITTI-360 Fisheye Data Loader)

### KittiData
A PyTorch dataset class designed for the KITTI-360 dataset with fisheye camera support.  
**GitHub:** [KittiData](https://your.repo.url/KittiData-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/KittiData.__init__-placeholder)  
**Purpose:**  
Initializes the dataset by:
- Setting environment paths (expects `KITTI360_DATASET` to be set).
- Loading sequences, poses, and annotations.
- Computing the BEV grid and additional transformation matrices.

**Parameters:**
- `is_train` (*bool*): Training mode flag.
- `data_aug_conf` (*dict*): Data augmentation parameters.
- `grid_conf` (*dict*): Grid configuration (boundaries and resolution).
- `is_aug` (*bool*, optional): Whether to use additional augmentation (default is `False`).

**Output:**  
- An instance of `KittiData`.

**Data Type:** Instance of `KittiData`

#### shift_origin
**GitHub:** [shift_origin](https://your.repo.url/KittiData.shift_origin-placeholder)  
**Purpose:**  
Applies an additional shift transformation to the BEV coordinate system.

**Parameters:**
- `x` (*float*, default `-0.81`): Shift along the X-axis.
- `y` (*float*, default `-0.32`): Shift along the Y-axis.
- `z` (*float*, default `-0.9`): Shift along the Z-axis.

**Output:**  
- *NumPy array:* 4×4 transformation matrix.

**Data Type:** `np.ndarray`

#### get_sequences
**GitHub:** [get_sequences](https://your.repo.url/KittiData.get_sequences-placeholder)  
**Purpose:**  
Retrieves and splits sequence names into training and validation sets.

**Parameters:**
- `val_idxs` (*List[int]*, optional): Indices to designate validation sequences (default `[8]`).

**Output:**  
- *List:* Sequence names for the current mode.

**Data Type:** `List[str]`

#### prepro
**GitHub:** [prepro](https://your.repo.url/KittiData.prepro-placeholder)  
**Purpose:**  
Processes sequences by loading poses from text files, aligning frames with poses, and packaging the data into a structured NumPy array.

**Parameters:**  
None.

**Output:**  
- *NumPy structured array:* Contains `sequence` (*str*), `frame` (*str*), and `pose` (4×4 flattened, *float32*).

**Data Type:** `np.ndarray`

#### get_bboxes
**GitHub:** [get_bboxes](https://your.repo.url/KittiData.get_bboxes-placeholder)  
**Purpose:**  
Loads 3D bounding box annotations for each sequence.

**Parameters:**
- `sequences` (*List[str]*): List of sequence names.

**Output:**  
- *Dictionary:* Mapping sequence names to their bounding box objects.

**Data Type:** `Dict[str, Any]`

#### sample_augmentation
**GitHub:** [sample_augmentation](https://your.repo.url/KittiData.sample_augmentation-placeholder)  
**Purpose:**  
Computes image augmentation parameters (resize, crop, flip, rotation) similar to `NuscData` but tailored for KITTI-360.

**Parameters:**  
None.

**Output:**  
- *Tuple:* `(resize, resize_dims, crop, flip, rotate)`

**Data Type:** `Tuple`

#### get_aug_image_data
**GitHub:** [get_aug_image_data](https://your.repo.url/KittiData.get_aug_image_data-placeholder)  
**Purpose:**  
Retrieves and augments image data using fisheye camera calibration parameters.

**Parameters:**
- `rec` (*dict*): A sample record.
- `cams` (*dict*): Dictionary of camera objects.

**Output:**  
- *Tuple:* Contains augmented images, rotation matrices, translation vectors, intrinsic parameters, and post-augmentation matrices.

**Data Type:** `Tuple[torch.Tensor, ...]`

#### get_image_data
**GitHub:** [get_image_data](https://your.repo.url/KittiData.get_image_data-placeholder)  
**Purpose:**  
Loads image data from the KITTI-360 dataset, handling fisheye distortions and camera projection.

**Parameters:**
- `rec` (*dict*): A sample record.
- `cams` (*dict*): Dictionary mapping camera identifiers to fisheye camera objects.

**Output:**  
- *Tuple:* `(imgs, rots, trans, intrinsics, distortions, xis)` where:
  - `imgs` (*torch.Tensor*): Normalized image tensors.
  - `rots` (*torch.Tensor*): Rotation matrices.
  - `trans` (*torch.Tensor*): Translation vectors.
  - `intrinsics` (*torch.Tensor*): Intrinsic parameter vectors.
  - `distortions` (*torch.Tensor*): Distortion coefficients.
  - `xis` (*torch.Tensor*): Fisheye-specific xi parameter.

**Data Type:** `Tuple[torch.Tensor, ...]`

#### get_lidar_data
**GitHub:** [get_lidar_data](https://your.repo.url/KittiData.get_lidar_data-placeholder)  
**Purpose:**  
Retrieves LiDAR data for a given sample.

**Parameters:**
- `rec` (*dict*): A sample record.
- `nsweeps` (*int*): Number of LiDAR sweeps.

**Output:**  
- *Tensor:* LiDAR point cloud (first 3 dimensions: x, y, z).

**Data Type:** `torch.Tensor`

#### get_binimg
**GitHub:** [get_binimg](https://your.repo.url/KittiData.get_binimg-placeholder)  
**Purpose:**  
Generates a binary BEV image by transforming 3D bounding boxes to the BEV plane.

**Parameters:**
- `rec` (*dict*): A sample record.

**Output:**  
- *Tensor:* Binary BEV image.

**Data Type:** `torch.Tensor`

#### get_cams
**GitHub:** [get_cams](https://your.repo.url/KittiData.get_cams-placeholder)  
**Purpose:**  
Selects the camera objects based on the configuration and whether augmentation is applied.

**Parameters:**  
None.

**Output:**  
- *Dictionary:* Mapping camera identifiers to fisheye camera objects.

**Data Type:** `Dict[str, Any]`

#### __str__
**GitHub:** [__str__](https://your.repo.url/KittiData.__str__-placeholder)  
**Purpose:**  
Returns a summary string representation of the dataset.

**Parameters:**  
None.

**Output:**  
- *String:* Summary including sample count, mode, and augmentation configuration.

**Data Type:** `str`

#### __len__
**GitHub:** [__len__](https://your.repo.url/KittiData.__len__-placeholder)  
**Purpose:**  
Returns the total number of samples.

**Parameters:**  
None.

**Output:**  
- *Integer:* Total sample count.

**Data Type:** `int`

---

### VizData (KITTI-360)
Specialized for visualization; extends `KittiData` and provides additional BEV map functionalities.  
**GitHub:** [VizData (KITTI-360)](https://your.repo.url/VizDataKitti-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/VizDataKitti.__init__-placeholder)  
**Purpose:**  
Initializes the visualization dataset, inheriting properties from `KittiData`.

**Parameters:**  
- Inherits all parameters from `KittiData.__init__`.

**Output:**  
- An instance of `VizData` for KITTI-360.

**Data Type:** Instance of `VizData`

#### get_colored_binimg
**GitHub:** [get_colored_binimg](https://your.repo.url/VizDataKitti.get_colored_binimg-placeholder)  
**Purpose:**  
Generates a colored BEV map by overlaying fisheye coverage, ego-vehicle visualization, and annotated objects.

**Parameters:**
- `rec` (*dict*): A sample record.
- `cams` (*dict*): Dictionary of camera objects.

**Output:**  
- *Tensor:* Colored BEV image.

**Data Type:** `torch.Tensor`

#### __getitem__
**GitHub:** [__getitem__](https://your.repo.url/VizDataKitti.__getitem__-placeholder)  
**Purpose:**  
Retrieves a complete sample for visualization including:
- Image data.
- Camera extrinsic/intrinsic parameters.
- Binary and colored BEV maps.

**Parameters:**
- `index` (*int*): Index of the sample.

**Output:**  
- *Tuple:* `(imgs, rots, trans, K, D, xi, binimg, colored_binimg)`

**Data Type:** `Tuple[torch.Tensor, ...]`

---

### SegmentationData (KITTI-360)
Specialized for segmentation tasks on KITTI-360 data; extends `KittiData`.  
**GitHub:** [SegmentationData (KITTI-360)](https://your.repo.url/SegmentationDataKitti-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/SegmentationDataKitti.__init__-placeholder)  
**Purpose:**  
Inherits initialization from `KittiData`.

**Parameters:**  
- Inherits all parameters from `KittiData.__init__`.

**Output:**  
- An instance of `SegmentationData`.

**Data Type:** Instance of `SegmentationData`

#### __getitem__
**GitHub:** [__getitem__](https://your.repo.url/SegmentationDataKitti.__getitem__-placeholder)  
**Purpose:**  
Retrieves a segmentation sample including image data and the binary BEV map.

**Parameters:**
- `index` (*int*): Index of the sample.

**Output:**  
- *Tuple:* `(imgs, rots, trans, K, D, xi, binimg)`

**Data Type:** `Tuple[torch.Tensor, ...]`

---

### worker_rnd_init (KITTI-360)
**GitHub:** [worker_rnd_init (KITTI-360)](https://your.repo.url/worker_rnd_initKitti-placeholder)  
**Purpose:**  
Initializes the random seed for KITTI-360 data loader workers.

**Parameters:**
- `x` (*int*): Worker index.

**Output:**  
- *Returns:* `None` (side-effect: sets NumPy random seed).

**Data Type:** `None`

---

### compile_data (KITTI-360)
**GitHub:** [compile_data (KITTI-360)](https://your.repo.url/compile_dataKitti-placeholder)  
**Purpose:**  
Compiles training and validation DataLoaders for the KITTI-360 fisheye data.

**Parameters:**
- `data_aug_conf` (*dict*): Data augmentation configuration.
- `grid_conf` (*dict*): BEV grid configuration.
- `is_aug` (*bool*): Flag indicating whether additional augmentation is used.
- `bsz` (*int*): Batch size.
- `nworkers` (*int*): Number of workers.
- `parser_name` (*str*): Specifies parser type (`'vizdata'` or `'segmentationdata'`).

**Output:**  
- *Tuple:* `(trainloader, valloader)` – PyTorch DataLoader instances.

**Data Type:** `Tuple[DataLoader, DataLoader]`

---

## Final Notes

- **Placeholder Text:**  
  Some descriptions are marked as "props" where specific details are not fully known. Please update these sections with exact descriptions based on your project requirements.

- **GitHub URLs:**  
  Replace all placeholder URLs (`https://your.repo.url/...`) with the correct links to the corresponding code locations in your repository.

This comprehensive documentation is intended as a handover guide to facilitate the understanding and further development of the modified data loaders for the LSS project.


# Lift-Plate-Shoot Model Documentation

This document provides detailed documentation for the model implementations in the Lift-Plate-Shoot (LSS) project. Two modules are described:

- **model.py** – The original model implementation for pinhole cameras.
- **fisheye_model.py** – The modified model implementation for fisheye cameras with additional methods to handle fisheye distortions.

> **Note:** All GitHub URLs are placeholders. Replace `https://your.repo.url/...` with your actual repository links.

---

## Table of Contents

- [Overview](#overview)
- [model.py Module (Pinhole Cameras)](#modelpy-module-pinhole-cameras)
  - [Up](#up)
    - [__init__](#up-init)
    - [forward](#up-forward)
  - [CamEncode](#camencode)
    - [__init__](#camencode-init)
    - [get_depth_dist](#camencode-get_depth_dist)
    - [get_depth_feat](#camencode-get_depth_feat)
    - [get_eff_depth](#camencode-get_eff_depth)
    - [forward](#camencode-forward)
  - [BevEncode](#bevencode)
    - [__init__](#bevencode-init)
    - [forward](#bevencode-forward)
  - [LiftSplatShoot](#liftsplatshoot)
    - [__init__](#liftsplatshoot-init)
    - [create_frustum](#liftsplatshoot-create_frustum)
    - [get_geometry](#liftsplatshoot-get_geometry)
    - [get_cam_feats](#liftsplatshoot-get_cam_feats)
    - [voxel_pooling](#liftsplatshoot-voxel_pooling)
    - [get_voxels](#liftsplatshoot-get_voxels)
    - [forward](#liftsplatshoot-forward)
  - [compile_model](#compile_model)
- [fisheye_model.py Module (Fisheye Cameras)](#fisheyemodelpy-module-fisheye-cameras)
  - [Up](#up-fisheye)
    - [__init__](#up-fisheye-init)
    - [forward](#up-fisheye-forward)
  - [CamEncode](#camencode-fisheye)
    - [__init__](#camencode-fisheye-init)
    - [get_depth_dist](#camencode-fisheye-get_depth_dist)
    - [get_depth_feat](#camencode-fisheye-get_depth_feat)
    - [get_eff_depth](#camencode-fisheye-get_eff_depth)
    - [forward](#camencode-fisheye-forward)
  - [BevEncode](#bevencode-fisheye)
    - [__init__](#bevencode-fisheye-init)
    - [forward](#bevencode-fisheye-forward)
  - [LiftSplatShoot](#liftsplatshoot-fisheye)
    - [__init__](#liftsplatshoot-fisheye-init)
    - [create_frustum](#liftsplatshoot-fisheye-create_frustum)
    - [undistort_points_pytorch](#liftsplatshoot-fisheye-undistort_points_pytorch)
    - [build_fisheye_circle_mask](#liftsplatshoot-fisheye-build_fisheye_circle_mask)
    - [get_geometry_fisheye](#liftsplatshoot-fisheye-get_geometry_fisheye)
    - [get_cam_feats](#liftsplatshoot-fisheye-get_cam_feats)
    - [voxel_pooling](#liftsplatshoot-fisheye-voxel_pooling)
    - [get_voxels](#liftsplatshoot-fisheye-get_voxels)
    - [forward](#liftsplatshoot-fisheye-forward)
  - [compile_model](#compile_model_fisheye)

---

## Overview

The original model (in **model.py**) is built for pinhole cameras and uses standard convolutional backbones (e.g., EfficientNet, ResNet18) to extract image features, project them to a BEV grid, and perform voxel pooling. The modified model in **fisheye_model.py** adapts these methods to handle fisheye distortions with additional functions for unprojection (using a differentiable MEI-model), mask building for fisheye circles, and geometry adjustments specific to fisheye cameras.

---

## model.py Module (Pinhole Cameras)

### Up  
A module for upsampling and feature fusion.  
**GitHub:** [Up](https://your.repo.url/Up-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/Up.__init__-placeholder)  
**Purpose:** Initializes the upsampling module with:
- A bilinear upsampling layer.
- Two convolutional blocks with BatchNorm and ReLU for feature refinement after concatenation.

**Parameters:**
- `in_channels` (*int*): Number of input channels.
- `out_channels` (*int*): Number of output channels.
- `scale_factor` (*int*, optional): Upsampling scale factor (default `2`).

**Output:**  
An instance of the `Up` module.  
**Data Type:** `nn.Module`

#### forward
**GitHub:** [forward](https://your.repo.url/Up.forward-placeholder)  
**Purpose:** Applies upsampling to `x1`, concatenates it with `x2`, and refines the result via convolution.

**Parameters:**
- `x1` (*torch.Tensor*): Feature map to be upsampled.
- `x2` (*torch.Tensor*): Feature map to be concatenated with upsampled `x1`.

**Output:**  
- *Tensor:* Output feature map after upsampling, concatenation, and convolution.  
**Data Type:** `torch.Tensor`

---

### CamEncode  
Encodes camera features and predicts depth distribution.  
**GitHub:** [CamEncode](https://your.repo.url/CamEncode-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/CamEncode.__init__-placeholder)  
**Purpose:** Initializes the camera encoder using a pretrained EfficientNet as backbone and prepares layers for upsampling and depth prediction.

**Parameters:**
- `D` (*int*): Depth channel count.
- `C` (*int*): Feature channel count.
- `downsample` (*int*): Downsampling factor.

**Output:**  
An instance of the `CamEncode` module.  
**Data Type:** `nn.Module`

#### get_depth_dist
**GitHub:** [get_depth_dist](https://your.repo.url/CamEncode.get_depth_dist-placeholder)  
**Purpose:** Computes a softmax distribution over depth predictions.

**Parameters:**
- `x` (*torch.Tensor*): Input tensor with raw depth scores.
- `eps` (*float*, optional): A small epsilon value to ensure numerical stability (default `1e-20`).

**Output:**  
- *Tensor:* Softmax-normalized depth distribution.  
**Data Type:** `torch.Tensor`

#### get_depth_feat
**GitHub:** [get_depth_feat](https://your.repo.url/CamEncode.get_depth_feat-placeholder)  
**Purpose:** Extracts depth features by obtaining effective depth maps and computing weighted feature maps.

**Parameters:**
- `x` (*torch.Tensor*): Input image tensor.

**Output:**  
- *Tuple:* `(depth, new_x)` where:
  - `depth` is the computed depth distribution.
  - `new_x` is the depth-weighted feature tensor.
  
**Data Type:** `Tuple[torch.Tensor, torch.Tensor]`

#### get_eff_depth
**GitHub:** [get_eff_depth](https://your.repo.url/CamEncode.get_eff_depth-placeholder)  
**Purpose:** Extracts intermediate features from EfficientNet, collecting endpoints and performing upsampling for depth prediction.

**Parameters:**
- `x` (*torch.Tensor*): Input image tensor.

**Output:**  
- *Tensor:* Upsampled feature map used for depth estimation.
  
**Data Type:** `torch.Tensor`

#### forward
**GitHub:** [forward](https://your.repo.url/CamEncode.forward-placeholder)  
**Purpose:** Passes the input through the depth feature pipeline to obtain refined camera features.

**Parameters:**
- `x` (*torch.Tensor*): Input image tensor.

**Output:**  
- *Tensor:* Camera feature map after depth encoding.
  
**Data Type:** `torch.Tensor`

---

### BevEncode  
Encodes BEV (Bird’s-Eye View) features from voxelized image features.  
**GitHub:** [BevEncode](https://your.repo.url/BevEncode-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/BevEncode.__init__-placeholder)  
**Purpose:** Initializes the BEV encoder using a ResNet18-based trunk with additional upsampling layers to produce the final BEV feature map.

**Parameters:**
- `inC` (*int*): Number of input channels.
- `outC` (*int*): Number of output channels.

**Output:**  
An instance of the `BevEncode` module.  
**Data Type:** `nn.Module`

#### forward
**GitHub:** [forward](https://your.repo.url/BevEncode.forward-placeholder)  
**Purpose:** Processes the input feature map through convolutional layers, residual blocks, and upsampling to generate BEV features.

**Parameters:**
- `x` (*torch.Tensor*): Input tensor.

**Output:**  
- *Tensor:* Final BEV feature map.
  
**Data Type:** `torch.Tensor`

---

### LiftSplatShoot  
The core model that lifts image features into 3D space, splats them onto a BEV grid, and aggregates features.  
**GitHub:** [LiftSplatShoot](https://your.repo.url/LiftSplatShoot-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/LiftSplatShoot.__init__-placeholder)  
**Purpose:** Initializes the model by setting grid parameters, preparing frustum generation, and instantiating camera and BEV encoders.

**Parameters:**
- `grid_conf` (*dict*): Configuration for grid boundaries and resolution.
- `data_aug_conf` (*dict*): Data augmentation configuration.
- `outC` (*int*): Number of output channels.
  
**Output:**  
An instance of the `LiftSplatShoot` model.
  
**Data Type:** `nn.Module`

#### create_frustum
**GitHub:** [create_frustum](https://your.repo.url/LiftSplatShoot.create_frustum-placeholder)  
**Purpose:** Generates a 3D frustum grid from image plane coordinates and depth candidates.

**Parameters:**  
None.

**Output:**  
- *Tensor:* Frustum grid of shape `[D, H, W, 3]` where D, H, W are determined by depth bounds and downsampling.
  
**Data Type:** `nn.Parameter`

#### get_geometry
**GitHub:** [get_geometry](https://your.repo.url/LiftSplatShoot.get_geometry-placeholder)  
**Purpose:** Computes the 3D geometry (x, y, z locations in the ego frame) for each pixel using camera rotations, translations, and intrinsic parameters.

**Parameters:**
- `rots` (*torch.Tensor*): Rotation matrices.
- `trans` (*torch.Tensor*): Translation vectors.
- `intrins` (*torch.Tensor*): Intrinsic matrices.
- `post_rots` (*torch.Tensor*): Post-augmentation rotation matrices.
- `post_trans` (*torch.Tensor*): Post-augmentation translation vectors.

**Output:**  
- *Tensor:* Geometry tensor of shape `[B, N, D, H/downsample, W/downsample, 3]`.
  
**Data Type:** `torch.Tensor`

#### get_cam_feats
**GitHub:** [get_cam_feats](https://your.repo.url/LiftSplatShoot.get_cam_feats-placeholder)  
**Purpose:** Extracts camera features using the camera encoder and reshapes them for voxel pooling.

**Parameters:**
- `x` (*torch.Tensor*): Input image tensor of shape `[B, N, C, imH, imW]`.

**Output:**  
- *Tensor:* Feature tensor of shape `[B, N, D, H/downsample, W/downsample, C]` (with channels permuted).
  
**Data Type:** `torch.Tensor`

#### voxel_pooling
**GitHub:** [voxel_pooling](https://your.repo.url/LiftSplatShoot.voxel_pooling-placeholder)  
**Purpose:** Aggregates features into voxels by pooling over 3D space using a “cumsum trick” or a custom autograd function.

**Parameters:**
- `geom_feats` (*torch.Tensor*): Geometry tensor.
- `x` (*torch.Tensor*): Camera feature tensor.

**Output:**  
- *Tensor:* Voxelized feature tensor reshaped to BEV grid.
  
**Data Type:** `torch.Tensor`

#### get_voxels
**GitHub:** [get_voxels](https://your.repo.url/LiftSplatShoot.get_voxels-placeholder)  
**Purpose:** Computes geometry from camera parameters, extracts camera features, and performs voxel pooling to produce a BEV representation.

**Parameters:**
- `x`, `rots`, `trans`, `intrins`, `post_rots`, `post_trans`: Corresponding camera parameters and image tensor.

**Output:**  
- *Tensor:* Voxelized feature map ready for BEV encoding.
  
**Data Type:** `torch.Tensor`

#### forward
**GitHub:** [forward](https://your.repo.url/LiftSplatShoot.forward-placeholder)  
**Purpose:** Runs the full pipeline:
1. Generates voxels from image features.
2. Encodes BEV features via the BEV encoder.

**Parameters:**
- `x`, `rots`, `trans`, `intrins`, `post_rots`, `post_trans`: Inputs as described above.

**Output:**  
- *Tensor:* Final output of the model (BEV feature map).
  
**Data Type:** `torch.Tensor`

---

### compile_model
**GitHub:** [compile_model](https://your.repo.url/compile_model-placeholder)  
**Purpose:**  
Helper function that instantiates and returns a `LiftSplatShoot` model.

**Parameters:**
- `grid_conf` (*dict*): Grid configuration.
- `data_aug_conf` (*dict*): Data augmentation configuration.
- `outC` (*int*): Number of output channels.

**Output:**  
- *Instance:* `LiftSplatShoot` model.
  
**Data Type:** `nn.Module`

---

## fisheye_model.py Module (Fisheye Cameras)

### Up  
Identical in structure to the pinhole version, used for upsampling in the fisheye model.  
**GitHub:** [Up](https://your.repo.url/Up-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/Up.__init__-placeholder)  
**Purpose:** Initializes upsampling and convolution modules.

**Parameters:**  
Same as in the pinhole model.

**Output:**  
Instance of `Up`.

#### forward
**GitHub:** [forward](https://your.repo.url/Up.forward-placeholder)  
**Purpose:** Upsamples, concatenates, and processes feature maps.

**Parameters:**  
- `x1`, `x2`: Input tensors.

**Output:**  
Upsampled and refined tensor.

---

### CamEncode  
Encodes camera features for fisheye images.  
**GitHub:** [CamEncode](https://your.repo.url/CamEncode-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/CamEncode.__init__-placeholder)  
**Purpose:** Initializes the fisheye camera encoder with a pretrained EfficientNet backbone and additional layers.

**Parameters:**  
Same as in the pinhole model.

**Output:**  
Instance of `CamEncode`.

#### get_depth_dist, get_depth_feat, get_eff_depth, forward
**GitHub:**  
- [get_depth_dist](https://your.repo.url/CamEncode.get_depth_dist-placeholder)  
- [get_depth_feat](https://your.repo.url/CamEncode.get_depth_feat-placeholder)  
- [get_eff_depth](https://your.repo.url/CamEncode.get_eff_depth-placeholder)  
- [forward](https://your.repo.url/CamEncode.forward-placeholder)  

**Purpose:**  
These methods function similarly to the pinhole version, computing depth distributions and encoding features adapted for fisheye imagery.

**Parameters & Outputs:**  
Same as described in the pinhole model, with any fisheye-specific adjustments noted in props.

---

### BevEncode  
Encodes BEV features from fisheye-processed voxels.  
**GitHub:** [BevEncode](https://your.repo.url/BevEncode-placeholder)

#### __init__ and forward
**GitHub:**  
- [__init__](https://your.repo.url/BevEncode.__init__-placeholder)  
- [forward](https://your.repo.url/BevEncode.forward-placeholder)  

**Purpose:**  
Same as the pinhole version.

**Parameters & Outputs:**  
As described in the pinhole model.

---

### LiftSplatShoot (Fisheye)  
A specialized model to handle fisheye camera distortions with additional methods.  
**GitHub:** [LiftSplatShoot (Fisheye)](https://your.repo.url/LiftSplatShoot-fisheye-placeholder)

#### __init__
**GitHub:** [__init__](https://your.repo.url/LiftSplatShoot.__init__-placeholder)  
**Purpose:**  
Initializes the fisheye model with grid parameters, creates the frustum, and instantiates camera and BEV encoders. Also sets an optional augmentation flag and initializes a mask variable.

**Parameters:**
- `grid_conf`, `data_aug_conf`, `outC`: Same as before.
- `is_aug` (*bool*, optional): Flag for additional augmentation (default `False`).

**Output:**  
Instance of `LiftSplatShoot` (Fisheye version).

#### create_frustum
**GitHub:** [create_frustum](https://your.repo.url/LiftSplatShoot.create_frustum-placeholder)  
**Purpose:**  
Creates a frustum grid from image coordinates and depth candidates.

**Parameters:**  
None.

**Output:**  
- *Tensor:* Frustum grid of shape `[D, H, W, 3]`.

#### undistort_points_pytorch
**GitHub:** [undistort_points_pytorch](https://your.repo.url/LiftSplatShoot.undistort_points_pytorch-placeholder)  
**Purpose:**  
Implements a differentiable MEI-model unprojection to undistort pixel coordinates.

**Parameters:**
- `points` (*torch.Tensor*): Distorted pixel coordinates of shape `(N, 2)`.
- `K` (*torch.Tensor*): Intrinsic parameters (shape `[3, 3]` or batched).
- `D` (*torch.Tensor*): Distortion coefficients (shape `[4,]`).
- `xi` (*torch.Tensor*): Mirror parameter.
- `iterations` (*int*, optional): Number of iterations (default `20`).

**Output:**  
- *Tensor:* Unit ray directions of shape `(N, 3)`.
  
**Data Type:** `torch.Tensor`

#### build_fisheye_circle_mask
**GitHub:** [build_fisheye_circle_mask](https://your.repo.url/LiftSplatShoot.build_fisheye_circle_mask-placeholder)  
**Purpose:**  
Generates a mask to exclude pixels outside the effective fisheye circle.

**Parameters:**
- `B` (*int*): Batch size.
- `N` (*int*): Number of cameras.
- `H`, `W` (*int*): Height and width of the feature map.
- `K` (*torch.Tensor*): Intrinsic parameters.

**Output:**  
- *Tensor:* Boolean mask of shape `[B, N, H, W]` (expanded appropriately).

**Data Type:** `torch.Tensor`

#### get_geometry_fisheye
**GitHub:** [get_geometry_fisheye](https://your.repo.url/LiftSplatShoot.get_geometry_fisheye-placeholder)  
**Purpose:**  
Computes the 3D geometry for fisheye images using the undistortion method and transforms points from camera to ego coordinates.

**Parameters:**
- `rots`, `trans` (*torch.Tensor*): Rotation and translation.
- `K`, `D`, `xi`: Fisheye-specific intrinsic, distortion, and mirror parameters.

**Output:**  
- *Tensor:* Geometry tensor of shape `[B, N, D, H_down, W_down, 3]` with points outside the effective circle masked.

**Data Type:** `torch.Tensor`

#### get_cam_feats, voxel_pooling, get_voxels
**GitHub:**  
- [get_cam_feats](https://your.repo.url/LiftSplatShoot.get_cam_feats-placeholder)  
- [voxel_pooling](https://your.repo.url/LiftSplatShoot.voxel_pooling-placeholder)  
- [get_voxels](https://your.repo.url/LiftSplatShoot.get_voxels-placeholder)  

**Purpose:**  
These methods function similarly to their pinhole counterparts but use the geometry from `get_geometry_fisheye` and apply the fisheye mask.

**Parameters & Outputs:**  
Same structure as described in the pinhole model.

#### forward
**GitHub:** [forward](https://your.repo.url/LiftSplatShoot.forward-placeholder)  
**Purpose:**  
Runs the full fisheye pipeline: obtains voxels using fisheye-specific geometry and encodes them into a BEV feature map.

**Parameters:**
- `x`, `rots`, `trans`, `intrins`, `post_rots`, `post_trans`: Input tensors and camera parameters.

**Output:**  
- *Tensor:* Final BEV feature map.
  
**Data Type:** `torch.Tensor`

---

### compile_model (Fisheye)
**GitHub:** [compile_model](https://your.repo.url/compile_model-fisheye-placeholder)  
**Purpose:**  
Helper function to instantiate and return a `LiftSplatShoot` model configured for fisheye cameras.

**Parameters:**
- `grid_conf` (*dict*): Grid configuration.
- `data_aug_conf` (*dict*): Data augmentation configuration.
- `outC` (*int*, optional): Number of output channels (default `1`).
- `is_aug` (*bool*, optional): Augmentation flag (default `False`).

**Output:**  
- *Instance:* `LiftSplatShoot` (Fisheye version).

**Data Type:** `nn.Module`

---

## Final Notes

- **Placeholder Text:**  
  Some descriptions use "props" where exact details may need updating. Please refine these sections based on your project specifications.

- **GitHub URLs:**  
  Replace all placeholder URLs (`https://your.repo.url/...`) with the actual links to the corresponding sections in your repository.

This documentation serves as a comprehensive guide for both the original and fisheye-adapted model implementations in the Lift-Plate-Shoot project.

# BEVSegmentation Module Documentation

This document provides detailed documentation for the **BEVSegmentation** module. This module contains configuration parameters and a collection of helper functions used to generate Bird’s-Eye View (BEV) segmentation maps from KITTI-360 data. These functions support tasks such as loading poses, color assignment, geometric transformations, and drawing on BEV maps.

> **Note:** All GitHub URLs are placeholders. Replace `https://your.repo.url/...` with your actual repository links.

---

## Table of Contents

- [Configuration](#configuration)
- [Helper Functions](#helper-functions)
  - [load_poses](#load_poses)
  - [assign_color](#assign_color)
  - [transform_points](#transform_points)
  - [world_to_bev_indices](#world_to_bev_indices)
  - [fill_polygon](#fill_polygon)
  - [get_bottom_face](#get_bottom_face)
  - [draw_ego_vehicle](#draw_ego_vehicle)
  - [draw_fisheye_coverage](#draw_fisheye_coverage)
- [Main Script (Optional)](#main-script-optional)

---

## Configuration

This section describes the global configuration parameters used for BEV segmentation.

- **bev_size**:  
  *Type:* `float`  
  *Description:* Size of the BEV area in meters (e.g., `20.0` for a 20m x 20m area).

- **bev_resolution**:  
  *Type:* `int`  
  *Description:* Resolution of the BEV grid (e.g., `200` for 200×200 pixels; 0.1 m per pixel).

- **bev_min** and **bev_max**:  
  *Type:* `float`  
  *Description:* The minimum and maximum coordinates in meters for the BEV grid. Computed as `-bev_size/2.0` and `bev_size/2.0`, respectively.

- **T_additional**:  
  *Type:* `np.ndarray` (4×4)  
  *Description:* A fixed transformation matrix that shifts the BEV plane relative to the IMU origin.  
  *Default Values:*  
  - X shift: `-0.81`  
  - Y shift: `-0.32`  
  - Z shift: `-0.9`

---

## Helper Functions

This module includes several helper functions for processing KITTI-360 data and generating BEV maps.

### load_poses
**GitHub:** [load_poses](https://your.repo.url/BEVSegmentation.load_poses-placeholder)

**Purpose:**  
Loads pose information from a file. Each line in the file is expected to contain 13 numbers: a frame index followed by 12 numbers representing a 3×4 IMU-to-world matrix.

**Parameters:**
- `filename` (*str*): Path to the pose file.
- `start` (*int*, optional): Number of lines to skip at the beginning (default `0`).
- `max_poses` (*int*, optional): Maximum number of poses to load (default `300`).

**Output:**  
- *NumPy array:* Loaded poses with data type `np.float32`.

---

### assign_color
**GitHub:** [assign_color](https://your.repo.url/BEVSegmentation.assign_color-placeholder)

**Purpose:**  
Assigns an RGB color based on a given global identifier, using the KITTI-360 label mappings.

**Parameters:**
- `globalId` (*int*): A unique global identifier for an object.

**Output:**  
- *Tuple:* `(R, G, B)` where each component is an integer (0–255).

---

### transform_points
**GitHub:** [transform_points](https://your.repo.url/BEVSegmentation.transform_points-placeholder)

**Purpose:**  
Transforms an array of 3D points using a given 4×4 transformation matrix.

**Parameters:**
- `points` (*np.ndarray*): Array of shape `(N, 3)` representing N 3D points.
- `T` (*np.ndarray*): A 4×4 transformation matrix.

**Output:**  
- *NumPy array:* Transformed points of shape `(N, 3)`.

---

### world_to_bev_indices
**GitHub:** [world_to_bev_indices](https://your.repo.url/BEVSegmentation.world_to_bev_indices-placeholder)

**Purpose:**  
Converts 2D world coordinates (in meters) into pixel indices on the BEV grid.

**Parameters:**
- `points_xy` (*np.ndarray*): 2D coordinates (in meters) with shape `(N, 2)`.
- `bev_min` (*float*): Minimum coordinate of the BEV grid.
- `bev_max` (*float*): Maximum coordinate of the BEV grid.
- `resolution` (*int*): Resolution of the BEV grid (number of pixels).

**Output:**  
- *NumPy array:* Pixel indices as an integer array of shape `(N, 2)`.

---

### fill_polygon
**GitHub:** [fill_polygon](https://your.repo.url/BEVSegmentation.fill_polygon-placeholder)

**Purpose:**  
Fills a polygon on a segmentation map with a specified RGB color.

**Parameters:**
- `seg_map` (*np.ndarray*): A segmentation map (RGB image) with shape `(H, W, 3)`.
- `polygon` (*list*): A list of (x, y) tuples representing polygon vertices.
- `color` (*tuple*, optional): An (R, G, B) tuple. If not provided, a default fill value of `1.0` is used.

**Output:**  
- *Side-effect:* The `seg_map` is modified in-place with the filled polygon.

---

### get_bottom_face
**GitHub:** [get_bottom_face](https://your.repo.url/BEVSegmentation.get_bottom_face-placeholder)

**Purpose:**  
Extracts the bottom face of a 3D bounding box by selecting the 4 vertices with the lowest Z-values in the IMU coordinate frame.

**Parameters:**
- `vertices_imu` (*np.ndarray*): Array of 3D vertices with shape `(N, 3)`.

**Output:**  
- *NumPy array:* An array of shape `(4, 3)` representing the 4 corners of the bottom face.  
- *Notes:* The function sorts vertices by their Z-coordinate and reorders them in a clockwise or counter-clockwise order based on the centroid.

---

### draw_ego_vehicle
**GitHub:** [draw_ego_vehicle](https://your.repo.url/BEVSegmentation.draw_ego_vehicle-placeholder)

**Purpose:**  
Draws a representation of the ego vehicle on the BEV map, including a green rectangle and an arrow.

**Parameters:**
- `bev_map` (*np.ndarray*): The BEV map image (RGB) with shape `(H, W, 3)`.

**Output:**  
- *Side-effect:* The BEV map is modified in-place with the ego vehicle drawn at the center.

---

### draw_fisheye_coverage
**GitHub:** [draw_fisheye_coverage](https://your.repo.url/BEVSegmentation.draw_fisheye_coverage-placeholder)

**Purpose:**  
Draws the coverage area of a fisheye camera on the BEV map. This involves:
1. Undistorting a set of predefined pixel coordinates.
2. Transforming these coordinates from the camera frame to the IMU frame.
3. Converting the transformed points into BEV pixel indices.
4. Filling the corresponding area on the BEV map with a specified color.

**Parameters:**
- `bev_map` (*np.ndarray*): The BEV map image (RGB) with shape `(H, W, 3)`.
- `bev_min` (*float*): Minimum BEV coordinate (in meters).
- `bev_max` (*float*): Maximum BEV coordinate (in meters).
- `bev_resolution` (*int*): Resolution of the BEV grid.
- `camera`: A `CameraFisheye` object containing camera parameters and projection methods.
- `color` (*tuple*, optional): An (R, G, B) tuple to fill the coverage area (default `(25, 25, 50)`).

**Output:**  
- *Side-effect:* The BEV map is modified in-place with the fisheye coverage area drawn.

---

## Main Script (Optional)

The module also contains a main script that demonstrates how to use the helper functions to:
- Load 3D bounding box annotations and poses.
- Create BEV maps.
- Draw ego vehicle and fisheye camera coverage.
- Process bounding boxes and generate segmentation maps.

This script is intended for testing and visualization purposes.

---

## Final Notes

- **Placeholder Text:**  
  Update any sections marked with "props" or where additional details are needed based on your project specifications.

- **GitHub URLs:**  
  Replace all placeholder URLs (`https://your.repo.url/...`) with the correct links to the corresponding functions in your repository.

This documentation serves as a comprehensive reference for the **BEVSegmentation** module, enabling developers to understand and extend its functionality.
