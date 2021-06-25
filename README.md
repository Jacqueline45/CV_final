# CV_final
## Modification
Modify from : https://github.com/biubug6/Pytorch_Retinaface  
- Model structure  
  -  New backbone
      - mobile net v1 0.25 (default) 
      - mobile net v1 0.5 
      - mobile net v2 
      - mobile net v3 small 
      - squeeze net  
  - More feature maps  
    -  3 (default)
    -  4 (for mobile net v2)
- Data augmentatation  
  - Add additional photometric transformation by imgaug
- Location loss
  - smooth l1 loss (default)
  - D-IoU loss
  - C-IoU loss
- Optimizer
  - SGD (default)  
  - Adam 
## Model size  
backbone | num of feature maps | model size (MB)
--------|:---------------------:|--------------
mobile net v1 0.25 | 3 | 1.7
mobile net v1 0.5 | 3| 4.2
mobile net v2 | 4 | 3.9 
mobile net v3 | 3 | 4.3

NOTE: layers of extracte feature maps and output channel size of convolution layers are defined in ```config.py```
## Usage
- train
```
python train.py --network ["mobile0.25" or "resnet50" or "squeezenet1_1_small" or "mbnetv3" or "mbnetv10.5" or "mbnetv2"] \
--run [experiment name] --save_dir [root dir for weights, loginfo]  \
--training_dataset [path to train/label.txt] --val_dataset [path to val/label.txt] \
--optim [Adam or SGD] --lr [lr] --momentum [momentum] \
-- type_loc ["L1" or "Diou" or "Ciou"] --aug [True for additional photometric augmentation] \
[--loadinfo_dir [path to dir that stores train_log.csv & val_log.csv]]  [--resume_net [model path]] [--resume_epoch [epoch]]
```
- test
```
python test_widerface.py --trained_model [model path] --network ["mobile0.25" or "resnet50" or "squeezenet1_1_small" or "mbnetv3" or "mbnetv10.5" or "mbnetv2"] --txt_pth [outut txt path] --mode [val or test]
```
The face_detection/ folder should be on the same folder level with this project folder.  
Generate solution text file.  
- evaluation
```
python3 setup.py build_ext --inplace
python3 evaluation.py  -p [path to solution txt] -g ./ground_truth/val_gt.pkl -lm
```
## Reproduce
- train
```
python train.py --network mobile0.25 --optim Adam --lr 1e-3 --type_loc L1 --aug False\
--run [experiment name] --save_dir [root dir for weights, loginfo]  \
--training_dataset [path to train/label.txt] --val_dataset [path to val/label.txt]
```
- test
```
python test_widerface.py --trained_model [model path] --network mobile0.25 --txt_pth [outut txt path] --mode [val or test]
```
NOTE: The face_detection/ folder should be on the same folder level with this project folder.  
## Usage (Team member)
First, git clone this repo (specify -b dev), and download dataset, toolkit on colab.
```
!gdown --id '12gyZX7DMlyPsSHtXalwB44xtkp1hu2H2' --output face_detection.zip
!unzip face_detection.zip
!git clone -b dev https://[GITHUB USERNAME]:[GITHUB PASSWORD]@github.com/Jacqueline45/CV_final.git
```
- train
```
% cd　/content/CV_final/
!CUDA_VISIBLE_DEVICES=0 python train.py --network ["mobile0.25" or "resnet50" or "squeezenet1_1_small" or "mbnetv3" or "mbnetv10.5" or "mbnetv2"] \
--run [experiment name] --save_dir [root dir for weights, loginfo]  \
--training_dataset [path to train/label.txt] --val_dataset [path to val/label.txt] \
--optim [Adam or SGD] --lr [lr] --momentum [momentum] \
-- type_loc ["L1" or "Diou" or "Ciou"] --aug [True for additional photometric augmentation] \
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
