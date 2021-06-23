from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50, cfg_snet, cfg_mbnetv3, cfg_mnet_0_5
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import numpy as np 
import pandas as pd

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='../../../face_detection/CV_dataset/train/label.txt', help='Training dataset directory')
parser.add_argument('--val_dataset', default='../../../face_detection/CV_dataset/val/label.txt', help='Validation dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50 or squeezenet1_1_small or mbnetv3 or mbnetv10.5')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_dir', default='../gdrive/MyDrive/CV_final_output/', help='root dir for weight and runs')
parser.add_argument('--run', default="tmp")
parser.add_argument('--loadinfo_dir', default=None)
parser.add_argument('--optim', default="SGD", help="SGD or Adam")

args = parser.parse_args()
print(args)
args.save_folder = os.path.join(args.save_dir, args.run, "weights")
args.saveinfo_pth = os.path.join(args.save_dir, args.run, "log")
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
if not os.path.exists(args.saveinfo_pth):
    os.makedirs(args.saveinfo_pth)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
elif args.network == "squeezenet1_1_small":
    cfg = cfg_snet
elif args.network == "mbnetv3":
    cfg = cfg_mbnetv3
elif args.network ==  "mbnetv10.5":
    cfg = cfg_mnet_0_5

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
val_dataset = args.val_dataset
validation_dataset = WiderFaceDetection(val_dataset, preproc(img_dim, rgb_mean))
save_folder = args.save_folder
save_epoch_step = 10

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)


if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

# load previous training info
if args.loadinfo_dir is not None:
    print("Loading training info ...")
    train_df = pd.read_csv(os.path.join(args.loadinfo_dir, "train_log.csv")) 
    val_df = pd.read_csv(os.path.join(args.loadinfo_dir, "val_log.csv"))
else:
    train_df = pd.DataFrame(columns=["iteration", "train_loss_loc", "train_loss_cls", "train_loss_landm", "train_loss"])
    val_df = pd.DataFrame(columns=["iteration", "val_loss_loc", "val_loss_cls", "val_loss_landm", "val_loss", "val_loss"])

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

if args.optim == "SGD":
  optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
elif args.optim == "Adam":
  optimizer = optim.Adam(net.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train(train_df, val_df):
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            # validation
            val_net = net
            val_loss_loc, val_loss_cls, val_loss_landm, val_loss = val(val_net, epoch)
            val_df = val_df.append({"iteration": iteration, "val_loss_loc": val_loss_loc, "val_loss_cls": val_loss_cls, 
                                    "val_loss_landm": val_loss_landm, "val_loss": val_loss}, ignore_index=True)
            # write training info to saveinfo_path
            train_df.to_csv(os.path.join(args.saveinfo_pth, "train_log.csv"), index=False)
            val_df.to_csv(os.path.join(args.saveinfo_pth, "val_log.csv"), index=False)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), os.path.join(save_folder, cfg['name']+ '_epoch_' + str(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        
        train_df = train_df.append({"iteration": iteration, "train_loss_loc": loss_l.item(), "train_loss_cls": loss_c.item(), 
                                    "train_loss_landm": loss_landm.item(), "train_loss": loss.item()}, ignore_index=True)

        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
 
    torch.save(net.state_dict(), os.path.join(save_folder, cfg['name'] + '_Final.pth'))
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

def val(net, epoch):
    net.eval()
    batch_iterator = iter(data.DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate))
    # load val data
    images, targets = next(batch_iterator)
    images = images.cuda()
    targets = [anno.cuda() for anno in targets]

    # forward
    with torch.no_grad():
        out = net(images)
    loss_l, loss_c, loss_landm = criterion(out, priors, targets)
    loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
    

    print('Validation Epoch:{} Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}'
            .format(epoch, loss_l.item(), loss_c.item(), loss_landm.item()))
    return loss_l.item(), loss_c.item(), loss_landm.item(), loss.item()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train(train_df, val_df)
