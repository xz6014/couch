# COUCH dataset (ECCV'22)
[[ArXiv]](http://arxiv.org/abs/2205.00541) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/couch)
<p align="center">
TBC
</p>
COUCH is a dataset for full-body human-chair interactions captured in various indoor environments. We provide multi-view RGBD frames, MoCap and corresponding 3D SMPL and chair models fits along with the annotated contacts between them.  

## Contents
1. [Dependencies](#dependencies)
2. [Dataset Structure](#dataset-structure)
3. [Example usage](#example-usage)
4. [License](#license)
5. [Citation](#citation)


## Dependencies
This repo relies on these external libraries:
1. psbody mesh library. See [installation](https://github.com/MPI-IS/mesh#installation). 
2. trimesh. `pip install trimesh`
3. igl. `conda install -c conda-forge igl`
4. pytorch3d. See [installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). We recommend using `pytorch3d<=0.5.0`, e.g. `conda install pytorch3d==0.5.0 -c pytorch3d`.


## Dataset Structure
After unzip the dataset, you can find three subfolders: `calibs`, `objects`, `sequences`. The summary of each folder is described below:
```
calibs: Kinect camera intrinsics and extrinsics for different locations
objects: 3D scans of the 20 objects
sequences: color, depth paired with SMPL and object fits of human-object interaction sequences
split.json: train and test split
```
We discuss details of each folder next:

**calibs**: This folder stores the calibrations of Kinects.

```
DATASET_PATH
|--calibs           # Kinect camera intrinsics and extrinsics for different locations
|----Date[xx]       # background and camera poses for the scene on this date
|------background   # background image and point cloud 
|------config       # camera poses
|---intrinsics      # intrinsics of 4 kinect camera
```

**objects**: This folder provides the scans of our template objects. 
```
DATASET_PATH
|--objects
|---object_name.ply  # simplified object mesh 
```

**sequences**: This folder provides multi-view RGB-D images and SMPL, object registrations.
```
DATASET_PATH
|--sequences
|----sequence_name
|------info.json  # a file storing the calibration information for the sequence
|------t*.000     # one frame folder
|--------k[0-3].color.jpg           # color images of the frame
|--------k[0-3].depth.png           # depth images 
|--------k[0-3].person_mask.jpg     # human masks
|--------k[0-3].obj_rend_mask.jpg   # object masks
|--------k[0-3].color.json          # openpose detections
|--------k[0-3].mocap.[json|ply]    # FrankMocap estimated pose and mesh
|--------person
|----------person.ply               # segmented person point cloud
|----------fit01                    # registered SMPL mesh and parameters
```
Note: we store the SMPL-H parameters and corresponding mesh inside each `fit01` folder. If you would like to use other body models e.g. SMPL or SMPL-X, please refer to [this repo](https://github.com/vchoutas/smplx/tree/master/transfer_model) for conversions between different body models. 

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. In total there are 231 sequences for training and 90 sequences for testing. 


## Example usage
Here we describe some example usages of our dataset: 


**Parse object pose parameters**

TBC

```
python tools/parse_obj_pose.py -s [the path to a COUCH sequence]
```
after runing this, you can see the transformed meshes stored under the folder `tmp/[sequence name]`. 

## License
Copyright (c) 2022 Xiaohan Zhang, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to **cite the COUCH: Towards Controllable Human-Chair Interactions** paper in documents and papers that report on research using this Software.

In case the images are used for publication or public presentations, you are required to <strong>blur all human faces</strong>.

## Citation
If you use our code or data, please cite:
```bibtex
@inproceedings{zhang22couch,
title = {COUCH: Towards Controllable Human-Chair Interactions},
author = {Zhang, Xiaohan and Bhatnagar, Bharat Lal and Guzov, Vladimir and Starke, Sebastian and Pons-Moll, Gerard},
booktitle = {European Conference on Computer Vision ({ECCV})},
month = {October},
organization = {{Springer}},
year = {2022}
}
```
