# Deep learning-based inverse kinematics

![alt text](./data/teaser.gif "Columns left-to-right, Left: 3d keypoints. Rigth: reconstructed animation")

## Introduction
This project presents a human inverse kinemtics solution based on deep learning. 
A graph convolution network is constructed to predict SMPLx joint angles from a tepmoral
sequence of relative 3d poses in COCO format.

## Install
```bash
conda env create -n motion python=3.8.2
conda activate motion
pip install -r requirements.txt
```

## prepare the dataset
- register and download [the Amass dataset](https://amass.is.tue.mpg.de/), save it to the folder AMASS_DIR

- register and download the [SMPLx](https://smpl-x.is.tue.mpg.de/) models, save it also to the folder AMASS_DIR

## Run test inference.
download the amass dataset, SMPLx models and save it to the folder AMASS_DIR
```bash
python inference.py ./data/sample_3d_poses/dance_contemporary.npz AMASS_DIR
```

# Train the model
```bash
python ./model_wrap.py --amass AMASS_DIR  --data_dir DIR_TO_SAVE_MODELS --smpl_mean ./data/smpl/smpl_mean_params.npz
```


