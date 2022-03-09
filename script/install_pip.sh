#!/bin/bash
set -e

pip install -r requirements.txt
cd lib
git clone https://github.com/mkocabas/multi-person-tracker.git
git clone https://github.com/nkolot/SPIN.git 
mv multi-person-tracker multi_person_tracker
rm -rf SPIN/data
ln -s data/base_data/spin_data
cd ..
