# CV_final
## Modification
Modify from : https://github.com/biubug6/Pytorch_Retinaface  
- ```train.py```: add tensorboard for train loss, **validation loss not yet written**  
- ```config.py```: pretrain, True->False
- ```test_widerface.py```: change read data way, **not yet parallelize (original version process each image one by one, which takes about 20 minutes on validation set)**
## Usage
First, git clone this repo (specify -b dev), and download dataset, toolkit on colab.
- train
```
% cd　/content/CV_final/
!CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 --training_dataset --save_folder
```
recommend saving weights to google drive
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
- draw learning curve
Place the loginfo in a file name log/.  
When executing this instruction again, there will be messages reminding you to kill past processes, just kill it by ```!kill PROCESS_NUM```.  
```
%load_ext tensorboard
%tensorboard --logdir_spec log
```
