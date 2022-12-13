# PoseRisk: Pose-based Risk Scoring

## Introduction  
This repository is the implementation of an automatic human hazard detection model based on a human posture. This project has collaborated with [SNU HIS LAB](http://his.snu.ac.kr/).  


<p align="center">
  <img src="asset/reba_example.gif" width="85%" />
</p>


## Install guidelines
We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. Install [PyTorch](https://pytorch.org/) >= 1.2 according to your GPU driver and Python >= 3.7.2.  
Install the requirements using conda:
```
sh script/install_conda.sh
```

## Preparations
We used [Simple Multi Person Tracker](https://github.com/mkocabas/multi-person-tracker) for video human detection, and [SPIN](https://github.com/nkolot/SPIN) for human pose estimation.
If these are not in `lib` directory after installation, put these two repository in `lib` directory.
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
|   |   |-- spin_data
```
- Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/downloads) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${ROOT}/data/base_data/human_models`.  
- Download SPIN data from [here](https://github.com/nkolot/SPIN). Please reference `Fetch data` section in the repository.
## Run code
We provide REBA and RULA score method. You should put some additional information for score estimation. Please refer `example/additional_information.json`.  

Most of the evaluation criteria referred from: [https://kosha.or.kr/kosha/business/musculoskeletal_c_d.do](https://kosha.or.kr/kosha/business/musculoskeletal_c_d.do)  
for REBA,
- `Legs_bilateral_weight_bearing/walking`: (1-2) If balanced of two legs 1, if not 2
- `Sitting`: (0-1) If seated 1, if not 0
- `Load/Force Score_L/R)`: (0-3) load amount, rapid build up of force, etc.
- `Arm_supported_leaning_L/R`: (0-1) If arm is supported or person is learning 1, if not 0 **(IMPORTANT)**
- `Coupling_L/R`: (0-3) well fitting handle, acceptable body part, etc.
- `Activity_Score_L/R`: (0-3) repeated small range actions large acitivity, etc.

for RULA,
- `Arm_supported_leaning_L/R`: (0-1) If arm is supported or person is learning 1, if not 0 **(IMPORTANT)**
- `A_Muscle_use_L/R`, `A_Load/Force_L/R`: (0-1, 0-3) load amount, rapid build up of force, etc.
- `Legs_bilateral_weight_bearing`: (1-2) If balanced of two legs 1, if not 2
- `B_Muscle_use`, `B_Load/Force` : (0-1, 0-3) well fitting handle, acceptable body part, etc.

Below is the running example.  

Example:
```
cd ${ROOT}
conda activate risk
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} 
```

## Debug mode
If want to get score sequences, you should use `debug` option.
```
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} --debug
```

If want to get pose sequences for several joints, you should add `debug_joints` option.
```
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} --debug --debug_joints {joint names for debugging}
```

If want to get estimation result only for one frame, you should add `debug_frame` option.
```
python main/run.py --type REBA,RULA --input {input video path} --info {additional information path} --output {output directory} --debug --debug_frame {id of frame}
```

## Reference
TBD
