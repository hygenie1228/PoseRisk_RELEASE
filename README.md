# HazardDetection with Pose estimation

## Introduction  
This repository is the implementation of an automatic human hazard detection model based on a human posture. This project has collaborated with [SNU HIS LAB](http://his.snu.ac.kr/).  
## Install guidelines
We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. Install [PyTorch](https://pytorch.org/) >= 1.2 according to your GPU driver and Python >= 3.7.2, and run `sh requirements.sh`. 
## Preparations
We used [Simple Multi Person Tracker](https://github.com/mkocabas/multi-person-tracker) for video human detection, and [SPIN](https://github.com/nkolot/SPIN) for human pose estimation.
Put these two repository in `lib` directory.
```
${ROOT}  
|-- lib  
|   |-- [multi-person-traker](https://github.com/mkocabas/multi-person-tracker)
|   |-- [SPIN](https://github.com/nkolot/SPIN)
```
The `data` directory structure should follow the below hierarchy.
```
${ROOT}  
|-- data  
|   |-- base_data  
|   |   |-- human_models  
|   |-- spin_data
```
- Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/downloads) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${ROOT}/data/base_data/human_models`.  
- Download SPIN data from [here](https://github.com/nkolot/SPIN). Please reference `Fetch data` section in the repository.
## Run code
We provide REBA and RULA score method. You should put some additional information for score estimation. Please refer `example/additional_information.json`.  

Write data according to the criteria below.
- `sitting`: if a person is sit 1, if not 0
- `arm_contact`: if an arm is supported 1, if not 0
- `feet_ground_contact`: if both feet are on the ground 1, if not 0
- `sitting_status`: evaluate sitting posture, good 1, worst 2
- `whole_load/force`, `arm_wrist_load/force`, `neck_body_leg_load/force`: evaluate load/force based on each criteria
- `handle_exist`: evaluate score related to the handle

Below is the running example.  

Example:
```
cd ${ROOT}
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} 
```

If want to debug only for one frame, you can get smpl model by using `debug_frame` option.
```
cd ${ROOT}
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} --debug_frame {id of frame}
```

## Reference
TBD
