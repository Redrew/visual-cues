# Installation
Install the following packages. Tested in python3.8

* pytorch
* tqdm

# Data
Download the files in https://drive.google.com/drive/folders/16dK8WYyTUsh4ijYi5IvZ8zIIb35G_LNF?usp=sharing to the dataset folder in this repository.

Specifically, we only need these files for our experiments
* dataset/nuscenes-train/labels.pkl
* dataset/nuscenes-val/labels.pkl
* dataset/center_features.pkl

To run the yolov7 experiments, we need to download the external rear signal dataset from http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal

# Implicit Visual Features from Lift Splat Shoot
We have already extracted the pretrained features and saved them in the file dataset/center_features.pkl

To regenerate the same file:
* Clone https://github.com/Redrew/lift-splat-shoot
* Follow the setup instructions in README.md
* Download the NuScenes dataset from https://www.nuscenes.org/nuscenes
* Run the command: `python main.py extract_model_preds trainval --modelf=./model525000.pt --dataroot=<NuScenes Dataset Path> --train-label-path="~/visual-cues/dataset/nuscenes-train/labels.pkl" --val-label-path="~/visual-cues/dataset/nuscenes-val/labels.pkl" --bsz=32`

# Run
Run the notebooks:
* implicit_visual_features.ipynb
* finetuned_yolov7_data_prep.ipynb
* finetuned_yolov7_train.ipynb
* zeroshot_detic.ipynb
* zeroshot_xclip.ipynb
