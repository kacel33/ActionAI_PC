## Introduction
Multi Person PoseEstimation By PyTorch를 수정하여 ActionAI코드와 결합하였습니다.  
ActionAI는 jetson보드에서만 실행할 수 있었는데 이 코드를 이용하면 PC에서도 실행이 가능합니다.

## Results

.

[![License](https://img.shields.io/github/license/smellslikeml/ActionAI)](https://opensource.org/licenses/MIT) 

## Require
1. [Pytorch](http://pytorch.org/)

## Installation
1. git submodule init && git submodule update

## Demo
- Download [converted pytorch model](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0).
- Compile the C++ postprocessing: `cd lib/pafprocess; sh make.sh` 
- swig -python -c++ pafprocess.i 
- python setup.py build_ext --inplace
- `python demo/picture_demo.py` to run the picture demo.
- `python demo/web_demo.py` to run the web demo.

## ActionAI 코드 실행
- python demo/actionai_demo.py --video {Video PATH}


Download link:
[rtpose](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0)

## Development environment

The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 4 1080ti GPU cards. Other platforms or GPU cards are not fully tested.   
저는 python3.7.10 on Ubuntu 18.04 RTX 3080에서 실행하였습니다. 

## Quick start

### 1. Preparation

#### 1.1 Prepare the dataset
- `cd training; bash getData.sh` to obtain the COCO 2017 images in `/data/root/coco/images/`, keypoints annotations in `/data/root/coco/annotations/`,
make them look like this:
```
${DATA_ROOT}
|-- coco
    |-- annotations
        |-- person_keypoints_train2017.json
        |-- person_keypoints_val2017.json
    |-- images
        |-- train2017
            |-- 000000000009.jpg
            |-- 000000000025.jpg
            |-- 000000000030.jpg
            |-- ... 
        |-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
        

```

### 2. How to train the model
- Modify the data directory in `train/train_VGG19.py` and `python train/train_VGG19.py`

## Related repository
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

### Network Architecture
- testing architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/pose.png)

- training architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/training_structure.png)


## Citation
Please cite the paper in your publications if it helps your research: 

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
# REFERENCE
https://github.com/smellslikeml/ActionAI, ActionAI   
https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation
