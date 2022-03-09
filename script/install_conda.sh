#!/bin/bash
set -e

export CONDA_ENV_NAME=risk
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

pip install -r requirements.txt
cd lib
if [ -d 'multi_person_tracker' ]; then
   echo "Already cloned multi_person_tracker"
else
    git clone https://github.com/mkocabas/multi-person-tracker.git
    mv multi-person-tracker multi_person_tracker
fi

if [ -d 'SPIN' ]; then
   echo "Already cloned SPIN"
else
    git clone https://github.com/nkolot/SPIN.git 
    rm -rf SPIN/data
    ln -s ../../data/base_data/spin_data SPIN/data
fi
cd ..
