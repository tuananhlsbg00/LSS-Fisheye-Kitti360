# LSS-Fisheye Documentation

This document provides detailed documentation for the modified in the Lift-Splat-Shoot-Fisheye project and briefly document the original scripts to highlight our works. The modifications enable support for Fisheye cameras using the KITTI-360 dataset, while retaining the original NuScenes data loader for legacy use.

The documentation covers five main modules:

- [**data.py**](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py) - The original implementation of map-style data loader for the NuScenes dataset.
- [**model.py**](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py) - The original implementation of LSS model for the NuScenes dataset.
- [**src_fisheye/data.py**](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py) - The modified implementation of map-style data loader for the KITTI-360 dataset with Fisheye cameras.
- [**src_fisheye/model.py**](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py) - The modified implementation LSS model for the KITTI-360 dataset with Fisheye cameras.
- [**src_fisheye/kitti360scripts/viewers/BEVSegmentation.py**](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py) - Helper fucntions to load Bird-eye-view segmentation ground truth.


> **Note:** For some Classes, Methods, or Functions which have minor or no change at all might not be mentioned, especially in modified data module and model module

---


<details>
  <summary>
	  <h1>Table of content</h1>
  </summary>

- [0. Overview](#0.)

- [1. NuScenes Data Loader (data.py)](#1.)

	- [1.1 NuscData](#1.1)
	
		- [1.1.1 \_\_init__](#1.1.1)

		- [1.1.2 fix_nuscenes_formatting](#1.1.2)

		- [1.1.3 get_scenes](#1.1.3)

		- [1.1.4 prepro](#1.1.4)

		- [1.1.5 sample_augmentation](#1.1.5)

		- [1.1.6 get_image_data](#1.1.6)

		- [1.1.7 get_lidar_data](#1.1.7)

		- [1.1.8 get_binimg](#1.1.8)

		- [1.1.9 choose_cams](#1.1.9)

		- [1.1.10 \_\_str__](#1.1.10)

		- [1.1.11 \_\_len__](#1.1.11)

	- [1.2 VizData](#1.2)
	
		- [1.2.1 \_\_init__](#1.2.1)

		- [1.2.2 \_\_getitem__](#1.2.2)

	- [1.3 SegmentationData](#1.3)

		- [1.3.1 \_\_init__](#1.3.1)

		- [1.3.2 \_\_getitem__](#1.3.2)

	- [1.4 worker_rnd_init](#1.4)

	- [1.5 compile_data](#1.5)

- [2. KITTI-360 Fisheye Data Loader(src_fisheye/data.py)](#2.)

	- [2.1 KittiData](#2.1)

		- [2.1.1\_\_init__](#2.1.1)

		- [2.1.2 shift_origin](#2.1.2)

		- [2.1.3 get_sequences](#2.1.3)

		- [2.1.4 prepro](#2.1.4)

		- [2.1.5 get_bboxes](#2.1.5)

		- [2.1.6 get_image_data](#2.1.6)

		- [2.1.7 get_binimg](#2.1.7)

	- [2.2 VizData (KITTI-360)](#2.2)
	
		- [2.2.1 \_\_init__](#2.2.1)

		- [2.2.2 get_colored_binimg](#2.2.2)

		- [2.2.3 \_\_getitem__](#2.2.3)

	- [2.3 SegmentationData (KITTI-360)](#2.3)

		- [2.3.1 \_\_init__](#2.3.1)

		- [2.3.2 \_\_getitem__](#2.3.2)

- [3. Original LSS model (models.py)](#3.)
		
	- [3.1 Up](#3.1)

		- [3.1.1 \_\_init__](#3.1.1)

		- [3.1.2 forward](#3.1.2)

		- [3.2 CamEncode](#3.2)

		- [3.2.1 \_\_init__](#3.2.1)

		- [3.2.2 get_depth_dist](#3.2.2)

		- [3.2.3 get_depth_feat](#3.2.3)

		- [3.2.4 get_eff_depth](#3.2.4)

		- [3.2.5 forward](#3.2.5)

	- [3.3 BevEncode](#3.3)

		- [3.3.1 \_\_init__](#3.3.1)

		- [3.3.2 forward](#3.3.2)

	- [3.4 LiftSplatShoot](#3.4)

		- [3.4.1 \_\_init__](#3.4.1)

		- [3.4.2 create_frustum](#3.4.2)

		- [3.4.3 get_geometry](#3.4.3)

		- [3.4.4 get_cam_feats](#3.4.4)

		- [3.4.5 voxel_pooling](#3.4.5)

		- [3.4.6 get_voxels](#3.4.6)

		- [3.4.7 forward](#3.4.7)

	- [3.5 compile_model](#3.5)

- [4. LSS-Fisheye (src_fisheye.py)](#4.)

	- [4.1 LiftSplatShoot](#4.1)

		- [4.1.1 \_\_init__](#4.1.1)

		- [4.1.2 undistort_points_pytorch](#4.1.3)

		- [4.1.3 build_fisheye_circle_mask](#4.1.4)

		- [4.1.4 get_geometry_fisheye](#4.1.5)

- [5. BEVSegmentation]()

	- [5.1 load_poses](#5.1)

	- [5.2 assign_color](#5.2)

	- [5.3 transform_points](#5.3)

	- [5.4 world_to_bev_indices](#5.4)

	- [5.5 fill_polygon](#5.5)

	- [5.6 get_bottom_face](#5.6)

	- [5.7 draw_ego_vehicle](#5.7)

	- [5.8 draw_fisheye_coverage](#5.8)
</details>

---

## <a  id="0."></a> 0. Overview

The modifications include:
- **Data Format Adaptation:**  
  The original `data.py` is tailored for the NuScenes dataset. The modified `data.py` supports the KITTI-360 format and Fisheye cameras, incorporating necessary changes in file paths, calibration and ditortion parameters, and annotations.
  
- **Camera & Annotation Handling:**  
  In the KITTI-360 version, new helper modules (e.g., `CameraFisheye`, `Annotation3D`) are used to handle camera models, fisheye distortions, and 3D annotations.

- **BEV Generation:**  
  Binary and colored Bird’s-Eye View (BEV) maps are generated using different strategies tailored to each dataset.
  
- **LSS-Fisheye model (main contribution)**
 Introduce methods to handle  MEI-model fisheye distortions and unprojection in parallel (ensure this step is differentiable), mask building to handle out-of-bound pixels, and geometry adjustments specific to fisheye cameras.
  
- **BEVSegmentation:**
 This module contains a collection of helper functions used to generate Bird’s-Eye View (BEV) segmentation maps from KITTI-360 data. These functions support tasks such as loading poses, color assignment, geometric transformations, and drawing on BEV maps.

---

## <a id="1."></a> 1. NuScenes Data Loader (Original)

### <a id="1.1"></a> 1.1 NuscData
A PyTorch map-style dataset class that loads and preprocesses data from the NuScenes dataset.  
**GitHub:** [NuscData](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L21-L208)

- #### <a id="1.1.1"></a> 1.1.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L21-L36)  
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

- #### <a id="1.1.2"></a> 1.1.2 fix_nuscenes_formatting
	**GitHub:** [fix_nuscenes_formatting](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L38-L45)  
	**Purpose:**  
	Ensures that file paths within the NuScenes object match the actual file locations.

	**Parameters:**  
	None.

	**Output:**  
	- *Side-effect:* Updates internal sample records with corrected file paths.  
	- *Returns:* `None`

- #### <a id="1.1.3"></a> 1.1.3 get_scenes
	**GitHub:** [get_scenes](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L73-L82)  
	**Purpose:**  
	Filters and retrieves scene names based on the training or validation split.

	**Parameters:**  
	None.

	**Output:**  
	* `List[str]`:* Scene names.  

- #### <a id="1.1.4"></a> 1.1.4 prepro
	**GitHub:** [prepro](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L84-L94)  
	**Purpose:**  
	Preprocesses the dataset by filtering samples belonging to the chosen scenes and sorting them.

	**Parameters:**  
	None.

	**Output:**  
	- *`List[dict]`:* Processed sample records.  

- #### <a id="1.1.5"></a> 1.1.5 sample_augmentation
	**GitHub:** [sample_augmentation](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L96-L119)  
	**Purpose:**  
	Computes augmentation parameters including resize factor, dimensions, crop coordinates, flip flag, and rotation angle.

	**Parameters:**  
	None.

	**Output:**  
	- *`Tuple`:* `(resize, resize_dims, crop, flip, rotate)` where:
	  - `resize` (*float*)
	  - `resize_dims` (*tuple of ints*): New dimensions after resizing.
	  - `crop` (*tuple of ints*): Crop coordinates.
	  - `flip` (*bool*): Indicates if horizontal flip is applied.
	  - `rotate` (*float*): Rotation angle.

- #### <a id="1.1.6"></a> 1.1.6 get_image_data
	**GitHub:** [get_image_data](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L121-L164)  
	**Purpose:**  
	Loads images from specified cameras, applies augmentation, and returns image tensors along with camera calibration data.

	**Parameters:**
	- `rec` (*dict*): A sample record from NuScenes.
	- `cams` (*List[str]*): List of camera identifiers.

	**Output:**  
	- *`Tuple[torch.Tensor, ...]`* Contains:
	  - `imgs` (*torch.Tensor*): Normalized image tensors.
	  - `rots` (*torch.Tensor*): Rotation matrices.
	  - `trans` (*torch.Tensor*): Translation vectors.
	  - `intrins` (*torch.Tensor*): Camera intrinsic matrices.
	  - `post_rots` (*torch.Tensor*): Post-augmentation rotation matrices.
	  - `post_trans` (*torch.Tensor*): Post-augmentation translation vectors.

- #### <a id="1.1.7"></a> 1.1.7 get_lidar_data
	**GitHub:** [get_lidar_data](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L166-L169)  
	**Purpose:**  
	Retrieves LiDAR point cloud data for the sample.

	**Parameters:**
	- `rec` (*dict*): A sample record.
	- `nsweeps` (*int*): Number of LiDAR sweeps to aggregate.

	**Output:**  
	- *`torch.Tensor`:* LiDAR points (first 3 dimensions: x, y, z).

- #### <a id="1.1.8"></a> 1.1.8 get_binimg
	**GitHub:** [get_binimg](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L171-L193)  
	**Purpose:**  
	Generates a binary BEV image by projecting object annotations onto the BEV grid.

	**Parameters:**
	- `rec` (*dict*): A sample record.

	**Output:**  
	- *`torch.Tensor`:* Binary BEV image with a shape compatible with the grid dimensions.

- #### <a id="1.1.9"></a> 1.1.9 choose_cams
	**GitHub:** [choose_cams](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L195-L201)  
	**Purpose:**  
	Randomly selects a subset of camera identifiers during training.

	**Parameters:**  
	None.

	**Output:**  
	- *`List[str]`:* Selected camera identifiers.

- #### <a id="1.1.10"></a> 1.1.10 \_\_str__
	**GitHub:** [\_\_str__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L203-L205)  
	**Purpose:**  
	Returns a string representation summarizing the dataset.

	**Parameters:**  
	None.

	**Output:**  
	
	* `str`:* Summary including the number of samples, split type, and augmentation configuration.

- #### <a id="1.1.11"></a> 1.1.11 \_\_len__
	**GitHub:** [\_\_len__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L207-L208)  
	**Purpose:**  
	Returns the number of samples in the dataset.

	**Parameters:**  
	None.

	**Output:**  
	- *`int`:* Total number of samples.

---

### <a id="1.2"></a> 1.2 VizData
Inherits from `NuscData` for visualization purposes.  
**GitHub:** [VizData](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L211-L223)

- #### <a id="1.2.1"></a> 1.2.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L212-L213)  
	**Purpose:**  
	Inherits and initializes all properties from `NuscData`.

	**Parameters:**  
	- Inherits all parameters from `NuscData.__init__`.

	**Output:**  
	- An instance of `VizData`.

- #### <a id="1.2.2"></a> 1.2.2 \_\_getitem__
	**GitHub:** [\_\_getitem__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L215-L223)  
	**Purpose:**  
	Retrieves a complete sample for visualization including image data, LiDAR data, and the binary BEV image.

	**Parameters:**
	- `index` (*int*): Index of the sample.

	**Output:**  
	- *`Tuple[torch.Tensor, ...]`:* `(imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg)` where:
	  - `imgs` (*torch.Tensor*): Augmented images.
	  - `rots` (*torch.Tensor*): Rotation matrices.
	  - `trans` (*torch.Tensor*): Translation vectors.
	  - `intrins` (*torch.Tensor*): Camera intrinsics.
	  - `post_rots` (*torch.Tensor*): Post-augmentation rotations.
	  - `post_trans` (*torch.Tensor*): Post-augmentation translations.
	  - `lidar_data` (*torch.Tensor*): LiDAR point cloud.
	  - `binimg` (*torch.Tensor*): Binary BEV image.

---

### <a id="1.3"></a> 1.3 SegmentationData
Specialized for segmentation tasks; inherits from `NuscData`.  
**GitHub:** [SegmentationData](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L226-L237)

- #### <a id="1.3.1"></a> 1.3.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L227-L228)  
	**Purpose:**  
	Inherits initialization from `NuscData`.

	**Parameters:**  
	- Inherits all parameters from `NuscData.__init__`.

	**Output:**  
	- An instance of `SegmentationData`.

- #### <a id="1.3.2"></a> 1.3.2 \_\_getitem__
	**GitHub:** [\_\_getitem__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L230-L237)  
	**Purpose:**  
	Retrieves a sample for segmentation tasks including images, calibration paramerters and the BEV binary ground truth.

	**Parameters:**
	- `index` (*int*): Index of the sample.

	**Output:**  
	- *Tuple:* `(imgs, rots, trans, intrins, post_rots, post_trans, binimg)`

---

### <a id="1.4"></a> 1.4 worker_rnd_init
**GitHub:** [worker_rnd_init](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L240-L241)  
**Purpose:**  
Initializes a random seed for data loader workers to ensure reproducibility.

**Parameters:**
- `x` (*int*): Worker index.

**Output:**  
- *Returns:* `None` (side-effect: sets the NumPy random seed).

---

### <a id="1.5"></a> 1.5 compile_data
**GitHub:** [compile_data](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py#L244-L267)  
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
- *`Tuple`:* `(trainloader, valloader)` – PyTorch DataLoader instances.

---

## <a id="2."></a> 2. KITTI-360 Fisheye Data Loader (src_fisheye/data.py)

### <a id="2.1"></a> 2.1 KittiData
A PyTorch dataset class designed for the KITTI-360 dataset with fisheye camera support.  
**GitHub:** [KittiData](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L27-L306)

- #### <a id="2.1.1"></a>  2.1.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L28-L57)  
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

- #### <a id="2.1.2"></a>  2.1.2 shift_origin
	**GitHub:** [shift_origin](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L60-L67)  
	**Purpose:**  
	Because the IMU/GPS module is not aligned with the center of the car so we applies an additional shift transformation to the BEV coordinate system.

	**Parameters:**
	- `x` (*float*, default `-0.81`): Shift along the X-axis.
	- `y` (*float*, default `-0.32`): Shift along the Y-axis.
	- `z` (*float*, default `-0.9`): Shift along the Z-axis.

	**Output:**  
	- * `np.ndarray`:* 4×4 transformation matrix.

- #### <a id="2.1.3"></a>  2.1.3 get_sequences
	**GitHub:** [get_sequences](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L69-L84)  
	**Purpose:**  
	Retrieves and splits sequence names into training and validation sets.

	**Parameters:**
	- `val_idxs` (*List[int]*, optional): Indices to designate validation sequences (default `[8]`).

	**Output:**  
	- *`List[str]`:* Sequence names for the current mode.

- #### <a id="2.1.4"></a>  2.1.4 prepro
	**GitHub:** [prepro](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L86-L111)  
	**Purpose:**  
	Processes sequences by loading poses from text files, aligning frames with poses, and packaging the data into a structured NumPy array.

	**Parameters:**  
	None.

	**Output:**  
	- *`np.ndarray`:* Contains:
		- `'sequence'` (*str*): Sequence name
		- `'frame'` (*str*): Timestamp which help to identify corresponding .png Fisheye images and dynamic bounding box.
		- `'pose'` (4×4 flattened, *np.float32*): Rigid body transform from GPU/IMU coordinates to a world coordinate system. The origin of this world coordinate system is the same for all sequences, chosen as the center of the sequences.


- #### <a id="2.1.5"></a>  2.1.5 get_bboxes
	**GitHub:** [get_bboxes](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L113-L122)  
	**Purpose:**  
	Loads 3D bounding box annotations for each sequence.

	**Parameters:**
	- `sequences` (*List[str]*): List of sequence names.

	**Output:**  
	- *`Dict[str, List[3D_BBox]]`*: A dictionary mapping sequence names (strings) to lists of 3D bounding box objects. Each 3D bounding box is represented as a `KITTI360Bbox3D` object with the following [attributes](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/annotation.py#L77-L100)

- #### <a id="2.1.6"></a>  2.1.6 get_image_data
	**GitHub:** [get_image_data](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L194-L242)  
	**Purpose:**  
	Loads image data from the KITTI-360 dataset, handling fisheye distortions and camera projection.

	**Parameters:**
	- `rec` (*dict*): A sample record.
	- `cams` (*dict*): Dictionary mapping camera identifiers to fisheye camera objects.

	**Output:**  
	- *`Tuple[torch.Tensor, ...]`:* `(imgs, rots, trans, intrinsics, distortions, xis)` where:
	  - `imgs` (*torch.Tensor*): Normalized image tensors.
	  - `rots` (*torch.Tensor*): Rotation matrices.
	  - `trans` (*torch.Tensor*): Translation vectors.
	  - `intrinsics` (*torch.Tensor*): Intrinsic parameter vectors.
	  - `distortions` (*torch.Tensor*): Distortion coefficients.
	  - `xis` (*torch.Tensor*): Fisheye MEI-Model xi parameter.

- #### <a id="2.1.7"></a>  2.1.7 get_binimg
	**GitHub:** [get_binimg](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L250-L287)  
	**Purpose:**  
	Generates a binary BEV image by transforming 3D bounding boxes to the BEV plane. Discard bounding boxes that not appear at `rec['frame']` timestamp.

	**Parameters:**
	- `rec` (*dict*): A sample record.

	**Output:**  
	- *`torch.Tensor`:* Binary BEV ground truth.

---

### <a id="2.2"></a> 2.2 VizData (KITTI-360)
Specialized for visualization; extends `KittiData` and provides additional BEV map functionalities.  
**GitHub:** [VizData (KITTI-360)](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L309-L382)

- #### <a id="2.2.1"></a> 2.2.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L310-L311)  
	**Purpose:**  
	Initializes the visualization dataset, inheriting properties from `KittiData`.

	**Parameters:**  
	- Inherits all parameters from `KittiData.__init__`.

	**Output:**  
	- An instance of `VizData` for KITTI-360.

- #### <a id="2.2.2"></a> 2.2.2 get_colored_binimg
	**GitHub:** [get_colored_binimg](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L313-L364)  
	**Purpose:**  
	Generates a colored BEV map by overlaying fisheye coverage, ego-vehicle visualization, and annotated objects.

	**Parameters:**
	- `rec` (*dict*): A sample record.
	- `cams` (*dict*): Dictionary of camera objects.

	**Output:**  
	- *`torch.Tensor`:* Colored BEV ground truth for better visualization.

- #### <a id="2.2.3"></a> 2.2.3 \_\_getitem__
	**GitHub:** [\_\_getitem__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L366-L382)  
	**Purpose:**  
	Retrieves a complete sample for visualization including:
	- Image data.
	- Camera extrinsic,intrinsic, and distortions parameters.
	- Binary and colored BEV maps.

	**Parameters:**
	- `index` (*int*): Index of the sample.

	**Output:**  
	- *`Tuple[torch.Tensor, ...]`:* `(imgs, rots, trans, K, D, xi, binimg, colored_binimg)` where: 
		- `imgs` (*torch.Tensor*): Normalized image tensors.
		- `rots` (*torch.Tensor*): Rotation matrices.
		- `trans` (*torch.Tensor*): Translation vectors.
		- `K` (*torch.Tensor*): Intrinsic parameter vectors.
		- `D` (*torch.Tensor*): Distortion coefficients.
		- `xi` (*torch.Tensor*): Fisheye MEI-Model xi parameter.
		- `binimg` (*torch.Tensor*): Binary BEV ground truth.
		- `colored_binimg` (*torch.Tensor*): Colored BEV map for better visualization

---

### <a id="2.3"></a> 2.3 SegmentationData (KITTI-360)
Specialized for segmentation tasks on KITTI-360 data; extends `KittiData`.  
**GitHub:** [SegmentationData (KITTI-360)](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L385-L404)

- #### <a id="2.3.1"></a> 2.3.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L386-L387)  
	**Purpose:**  
	Inherits initialization from `KittiData`.

	**Parameters:**  
	- Inherits all parameters from `KittiData.__init__`.

	**Output:**  
	- An instance of `SegmentationData`.

- #### <a id="2.3.2"></a> 2.3.2 \_\_getitem__
	**GitHub:** [\_\_getitem__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/data.py#L389-L404)  
	**Purpose:**  
	Retrieves a segmentation sample including image data and the binary BEV map.

	**Parameters:**
	- `index` (*int*): Index of the sample.

	**Output:**  
	- *`Tuple[torch.Tensor, ...]`:* `(imgs, rots, trans, K, D, xi, binimg)` where: 
		- `imgs` (*torch.Tensor*): Normalized image tensors.
		- `rots` (*torch.Tensor*): Rotation matrices.
		- `trans` (*torch.Tensor*): Translation vectors.
		- `K` (*torch.Tensor*): Intrinsic parameter vectors.
		- `D` (*torch.Tensor*): Distortion coefficients.
		- `xi` (*torch.Tensor*): Fisheye MEI-Model xi parameter.
		- `binimg` (*torch.Tensor*): Binary BEV ground truth.

---

## <a id="3."></a> 3. Original LSS model (model.py)

### <a id="3.1"></a>  3.1 Up
A module for upsampling and feature fusion.  
**GitHub:** [Up](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L15-L34)

- #### <a id="3.1.1"></a> 3.1.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L16-L29)  
	**Purpose:** Initializes the upsampling module with:
	- A bilinear upsampling layer.
	- Two convolutional blocks with BatchNorm and ReLU for feature refinement after concatenation.

	**Parameters:**
	- `in_channels` (*int*): Number of input channels.
	- `out_channels` (*int*): Number of output channels.
	- `scale_factor` (*int*, optional): Upsampling scale factor (default `2`).

	**Output:**  
	An instance of the `Up` module. 

- #### <a id="3.1.2"></a> 3.1.2 forward
	**GitHub:** [forward](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L31-L34)  
	**Purpose:** Applies upsampling to `x1`, concatenates it with `x2`, and refines the result via convolution.

	**Parameters:**
	- `x1` (*torch.Tensor*): Feature map to be upsampled.
	- `x2` (*torch.Tensor*): Feature map to be concatenated with upsampled `x1`.

	**Output:**  
	- *`torch.Tensor`:* Output feature map after upsampling, concatenation, and convolution.  

---

### <a id="3.2"></a> 3.2 CamEncode
Encodes camera features and predicts depth distribution.  
**GitHub:** [CamEncode](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L37-L87)

- #### <a id="3.2.1"></a> 3.2.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L38-L46)  
	**Purpose:** Initializes the camera encoder using a pretrained EfficientNet as backbone and prepares layers for upsampling and depth prediction.

	**Parameters:**
	- `D` (*int*): Depth channel count.
	- `C` (*int*): Feature channel count.
	- `downsample` (*int*): Downsampling factor.

	**Output:**  
	An instance of the `CamEncode` module.  

- #### <a id="3.2.2"></a> 3.2.2 get_depth_dist
	**GitHub:** [get_depth_dist](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L48-L49)  
	**Purpose:** Computes a softmax distribution over depth predictions.

	**Parameters:**
	- `x` (*torch.Tensor*): Input tensor with raw depth scores.
	- `eps` (*float*, optional): A small epsilon value to ensure numerical stability (default `1e-20`).

	**Output:**  
	- *`torch.Tensor`:* Softmax-normalized depth distribution.  

- #### <a id="3.2.3"></a> 3.2.3 get_depth_feat
	**GitHub:** [get_depth_feat](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L51-L59)  
	**Purpose:** Extracts depth features by obtaining effective depth maps and computing weighted feature maps.

	**Parameters:**
	- `x` (*torch.Tensor*): Input image tensor.

	**Output:**  
	- *`Tuple[torch.Tensor, torch.Tensor]`:* `(depth, new_x)` where:
	  - `depth` is the computed depth distribution.
	  - `new_x` is the depth-weighted feature tensor.

* #### <a id="3.2.4"></a> 3.2.4 get_eff_depth
	**GitHub:** [get_eff_depth](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L61-L82)  
	**Purpose:** Extracts intermediate features from EfficientNet, collecting endpoints and performing upsampling for depth prediction.

	**Parameters:**
	- `x` (*torch.Tensor*): Input image tensor.

	**Output:**  
	- *`torch.Tensor`:* Upsampled feature map used for depth estimation.
	  


- #### <a id="3.2.5"></a> 3.2.5 forward
	**GitHub:** [forward](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L84-L87)  
	**Purpose:** Passes the input through the depth feature pipeline to obtain refined camera features.

	**Parameters:**
	- `x` (*torch.Tensor*): Input image tensor.

	**Output:**  
	- *`torch.Tensor`:* Camera feature map after depth encoding.

---

### <a id="3.3"></a> 3.3 BevEncode
Encodes BEV (Bird’s-Eye View) features from voxelized image features.  
**GitHub:** [BevEncode](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L90-L126)

- #### <a id="3.3.1"></a> 3.3.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L91-L112)  
	**Purpose:** Initializes the BEV encoder using a ResNet18-based trunk with additional upsampling layers to produce the final BEV feature map.

	**Parameters:**
	- `inC` (*int*): Number of input channels.
	- `outC` (*int*): Number of output channels.

	**Output:**  
	An instance of the `BevEncode` module.  

- #### <a id="3.3.2"></a> 3.3.2 forward
	**GitHub:** [forward](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L114-L126)  
	**Purpose:** Processes the input feature map through convolutional layers, residual blocks, and upsampling to generate BEV features.

	**Parameters:**
	- `x` (*torch.Tensor*): Input tensor.

	**Output:**  
	- *`torch.Tensor`:* Final BEV feature map.

---

### <a id="3.4"></a> 3.4 LiftSplatShoot
The core model that lifts image features into 3D space, splats them onto a BEV grid, and aggregates features.  
**GitHub:** [LiftSplatShoot](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L129-L255)

- #### <a id="3.4.1"></a> 3.4.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L130-L151)  
	**Purpose:** Initializes the model by setting grid parameters, preparing frustum generation, and instantiating camera and BEV encoders.

	**Parameters:**
	- `grid_conf` (*dict*): Configuration for grid boundaries and resolution.
	- `data_aug_conf` (*dict*): Data augmentation configuration.
	- `outC` (*int*): Number of output channels.
	  
	**Output:**  
	An instance of the `LiftSplatShoot` model.

- #### <a id="3.4.2"></a> 3.4.2 create_frustum
	**GitHub:** [create_frustum](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L153-L164)  
	**Purpose:** Generates a 3D frustum grid from image plane coordinates and depth candidates. 

	**Parameters:**  
	None.

	**Output:**  
	- *`nn.Parameter`:* Frustum grid of shape `[D, H, W, 3]` where D, H, W are determined by depth bounds and downsampling.

- #### <a id="3.4.3"></a> 3.4.3 get_geometry
	**GitHub:** [get_geometry](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L166-L186)  
	**Purpose:** Computes the 3D geometry (x, y, z locations in the ego frame) for each pixel using camera rotations, translations, and intrinsic parameters.

	**Parameters:**
	- `rots` (*torch.Tensor*): Rotation matrices.
	- `trans` (*torch.Tensor*): Translation vectors.
	- `intrins` (*torch.Tensor*): Intrinsic matrices.
	- `post_rots` (*torch.Tensor*): Post-augmentation rotation matrices.
	- `post_trans` (*torch.Tensor*): Post-augmentation translation vectors.

	**Output:**  
	- *`torch.Tensor`:* Geometry tensor of shape `[B, N, D, H/downsample, W/downsample, 3]`.

- #### <a id="3.4.4"></a> 3.4.4 get_cam_feats
	**GitHub:** [get_cam_feats](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L188-L198)  
	**Purpose:** Extracts camera features using the camera encoder and reshapes them for voxel pooling.

	**Parameters:**
	- `x` (*torch.Tensor*): Input image tensor of shape `[B, N, C, imH, imW]`.

	**Output:**  
	- *`torch.Tensor`:* Feature tensor of shape `[B, N, D, H/downsample, W/downsample, C]` (with channels permuted).

- #### <a id="3.4.5"></a> 3.4.5 voxel_pooling
	**GitHub:** [voxel_pooling](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200-L242)  
	**Purpose:** Aggregates features into voxels by pooling over 3D space using a “cumsum trick” or a custom autograd function.

	**Parameters:**
	- `geom_feats` (*torch.Tensor*): Geometry tensor.
	- `x` (*torch.Tensor*): Camera feature tensor.

	**Output:**  
	- *`torch.Tensor`:* Voxelized feature tensor reshaped to BEV grid.

- #### <a id="3.4.6"></a> 3.4.6 get_voxels
	**GitHub:** [get_voxels](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L244-L250)  
	**Purpose:** Computes geometry from camera parameters, extracts camera features, and performs voxel pooling to produce a BEV representation.

	**Parameters:**
	- `x`, `rots`, `trans`, `intrins`, `post_rots`, `post_trans`: Corresponding camera parameters and image tensor.

	**Output:**  
	- *`torch.Tensor`:* Voxelized feature map ready for BEV encoding.

- #### <a id="3.4.7"></a> 3.4.7 forward
	**GitHub:** [forward](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L252-L255)  
	**Purpose:** Runs the full pipeline:
	1. Generates voxels from image features.
	2. Encodes BEV features via the BEV encoder.

	**Parameters:**
	- `x`, `rots`, `trans`, `intrins`, `post_rots`, `post_trans`: Inputs as described above.

	**Output:**  
	- *`torch.Tensor`:* Final output of the model (BEV feature map).

---

### <a id="3.5"></a>  3.5 compile_model
**GitHub:** [compile_model](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L258-L259)  
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

## <a id="4."></a> 4. LSS-Fisheye model (src_fisheye/models.py)

- ### <a id="4.1"></a> 4.1 LiftSplatShoot
	A specialized model to handle fisheye camera distortions with additional methods.  
	**GitHub:** [LiftSplatShoot (Fisheye)](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py#L129-L426)

	#### <a id="4.1.1"></a> 4.1.1 \_\_init__
	**GitHub:** [\_\_init__](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py#L130-L156)  
	**Purpose:**  
	Initializes the fisheye model with grid parameters, creates the frustum, and instantiates camera and BEV encoders. Also sets an optional augmentation flag and initializes a mask variable.

	**Parameters:**
	- `grid_conf`, `data_aug_conf`, `outC`: Same as before.
	- `is_aug` (*bool*, optional): Flag for additional augmentation (default `False`).

	**Output:**  
	Instance of `LiftSplatShoot` (Fisheye version).

- #### <a id="4.1.3"></a> 4.1.2 undistort_points_pytorch
	**GitHub:** [undistort_points_pytorch](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py#L194-L251)  
	**Purpose:**  
	Implements a differentiable MEI-model unprojection to undistort pixel to 3D coordinate.

	**Parameters:**
	- `points` (*torch.Tensor*): Distorted pixel coordinates of shape `(N, 2)`.
	- `K` (*torch.Tensor*): Intrinsic parameters (shape `[3, 3]` or batched).
	- `D` (*torch.Tensor*): Distortion coefficients (shape `[4,]`).
	- `xi` (*torch.Tensor*): Mirror parameter.
	- `iterations` (*int*, optional): Number of iterations (default `20`).

	**Output:**  
	- *`torch.Tensor`:* Unit ray directions of shape `(N, 3)`.

- #### <a id="4.1.4"></a> 4.1.3 build_fisheye_circle_mask
	**GitHub:** [build_fisheye_circle_mask](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py#L253-L301)  
	**Purpose:**  
	Generates a mask to exclude out-of-bound pixels which lie outside the effective visible zone on the sensor.

	**Parameters:**
	- `B` (*int*): Batch size.
	- `N` (*int*): Number of cameras.
	- `H`, `W` (*int*): Height and width of the feature map.
	- `K` (*torch.Tensor*): Intrinsic parameters.

	**Output:**  
	- *`torch.Tensor`:* Boolean mask of shape `[B, N, H, W]` (expanded appropriately).

- #### <a id="4.1.5"></a> 4.1.4 get_geometry_fisheye
	**GitHub:** [get_geometry_fisheye](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/models.py#L303-L355)  
	**Purpose:**  
	Computes the 3D geometry for fisheye images using the undistortion method and transforms points from camera to ego coordinates. Then masking out all out-of-bound points

	**Parameters:**
	- `rots`, `trans` (*torch.Tensor*): Rotation and translation.
	- `K`, `D`, `xi`: Fisheye-specific intrinsic, distortion, and mirror parameters.

	**Output:**  
	- *`torch.Tensor`:* Geometry tensor of shape `[B, N, D, H_down, W_down, 3]` with points outside the effective circle masked.

---
## <a id="5."></a>  5. BEVSegmentation

This module includes several helper functions for processing KITTI-360 data and generating BEV maps.

###  <a id="5.1"></a>  5.1 load_poses
**GitHub:** [load_poses](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L41-L48)

**Purpose:**  
Loads pose information from a file. Each line in the file is expected to contain 13 numbers: a frame index followed by 12 numbers representing a 3×4 IMU-to-world matrix.

**Parameters:**
- `filename` (*str*): Path to the pose file.
- `start` (*int*, optional): Number of lines to skip at the beginning (default `0`).
- `max_poses` (*int*, optional): Maximum number of poses to load (default `300`).

**Output:**  
- *`np.Array`:* Loaded poses with data type `np.float32`.

---

### <a id="5.2"></a>  5.2 assign_color
**GitHub:** [assign_color](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L50-L61)

**Purpose:**  
Assigns an RGB color based on a given global identifier, using the KITTI-360 label mappings.

**Parameters:**
- `globalId` (*int*): A unique global identifier for an object.

**Output:**  
- *`Tuple`:* `(R, G, B)` where each component is an integer (0–255).

---

### <a id="5.3"></a>  5.3 transform_points
**GitHub:** [transform_points](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L63-L70)

**Purpose:**  
Transforms an array of 3D points using a given 4×4 transformation matrix.

**Parameters:**
- `points` (*np.ndarray*): Array of shape `(N, 3)` representing N 3D points.
- `T` (*np.ndarray*): A 4×4 transformation matrix.

**Output:**  
- *NumPy array:* Transformed points of shape `(N, 3)`.

---

### <a id="5.4"></a>  5.4 world_to_bev_indices
**GitHub:** [world_to_bev_indices](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L73-L79)

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

###  <a id="5.5"></a>  5.5 fill_polygon
**GitHub:** [fill_polygon](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L82-L98)

**Purpose:**  
Fills a polygon on a segmentation map with a specified RGB color.

**Parameters:**
- `seg_map` (*np.ndarray*): A segmentation map (RGB image) with shape `(H, W, 3)`.
- `polygon` (*list*): A list of (x, y) tuples representing polygon vertices.
- `color` (*tuple*, optional): An (R, G, B) tuple. If not provided, a default fill value of `1.0` is used.

**Output:**  
- *Side-effect:* The `seg_map` is modified in-place with the filled polygon.

---

### <a id="5.6"></a>  5.6 get_bottom_face
**GitHub:** [get_bottom_face](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L102-L124)

**Purpose:**  
Extracts the bottom face of a 3D bounding box by selecting the 4 vertices with the lowest Z-values in the IMU coordinate frame.

**Parameters:**
- `vertices_imu` (*np.ndarray*): Array of 3D vertices with shape `(N, 3)`.

**Output:**  
- *NumPy array:* An array of shape `(4, 3)` representing the 4 corners of the bottom face.  
- *Notes:* The function sorts vertices by their Z-coordinate and reorders them in a clockwise or counter-clockwise order based on the centroid.

---

### <a id="5.7"></a>  5.7 draw_ego_vehicle
**GitHub:** [draw_ego_vehicle](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L126-L155)

**Purpose:**  
Draws a representation of the ego vehicle on the BEV map, including a green rectangle and an arrow.

**Parameters:**
- `bev_map` (*np.ndarray*): The BEV map image (RGB) with shape `(H, W, 3)`.

**Output:**  
- *Side-effect:* The BEV map is modified in-place with the ego vehicle drawn at the center.

---

### <a id="5.8"></a>  5.8 draw_fisheye_coverage
**GitHub:** [draw_fisheye_coverage](https://github.com/tuananhlsbg00/LSS-Fisheye-Kitti360/blob/main/src_fisheye/kitti360scripts/viewer/BEVSegmentation.py#L158-L190)

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
