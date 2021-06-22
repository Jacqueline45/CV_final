# CV_final
## Modification
Modify from : https://github.com/biubug6/Pytorch_Retinaface  
- ```train.py```: add tensorboard for train loss, **validation loss not yet written**  
- ```config.py```: pretrain, True->False
- ```test_widerface.py```: change read data way, **not yet parallelize (original version process each image one by one, I think the time is acceptable when the model detects reasonable number of faces)**
## Usage
First, git clone this repo (specify -b dev), and download dataset, toolkit on colab.
```
!gdown --id '12gyZX7DMlyPsSHtXalwB44xtkp1hu2H2' --output face_detection.zip
!unzip face_detection.zip
!git clone -b dev https://[GITHUB USERNAME]:[GITHUB PASSWORD]@github.com/Jacqueline45/CV_final.git
```
- train
```
% cd　/content/CV_final/
!CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 --run [experiment name] --save_dir [root dir for weights, loginfo]  \
--training_dataset [path to train/label.txt] --val_dataset [path to val/label.txt] \
[--loadinfo_dir [path to dir that stores train_log.csv & val_log.csv]]  [--resume_net [model path]] [--resume_epoch [epoch]]
```
Recommend saving weights to google drive, i.e. save_dir should be dir in google drive  
Use loadinfo_dir, resume_net, and resume_epoch to resume training, but the tensorboard event will be a new separate file.  
The log files will be saved as SAVE_DIR/RUN/log/train_log.csv and SAVE_DIR/RUN/log/val_log.csv.  
The weights will be saved in SAVE_DIR/RUN
- test
```
% cd /content/CV_final/
!python test_widerface.py --trained_model ../mobilenet0.25_epoch_10.pth --network mobile0.25 --txt_pth "./solution.txt" --mode "test"
```
mode = "test" or "val"
- evaluate
```
% cd　/content/face_detection/eval_toolkit
!python3 setup.py build_ext --inplace
!python3 evaluation.py  -p ./solution.txt -g ./ground_truth/val_gt.pkl -lm
```
- Draw learning curve  
```
% cd /content/
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
train_df = pd.read_csv('train_log.csv')
val_df = pd.read_csv('val_log.csv')
train_cols = ['iteration', 'train_loss_loc', 'train_loss_cls', 'train_loss_landm','train_loss']
val_cols = ['iteration', 'val_loss_loc', 'val_loss_cls', 'val_loss_landm', 'val_loss', 'val_loss.1']
plt.plot(np.array(val_df['iteration']), np.array(val_df['val_loss'])) # change df (train/val) and the column name you want to draw
plt.show()
```
