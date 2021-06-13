import argparse
import torchvision.models as models
import torch

from models.retinaface import RetinaFace
from models.config import cfg_re50, cfg_mnet


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet")

model = RetinaFace(cfg_mnet, 'train')
x = torch.zeros([1, 3, 64, 64], dtype=torch.float32)
y = model(x)
print(y[0].shape) # [1, 168, 4]
print(y[1].shape) # [1, 168, 2]
print(y[2].shape) # [1, 168, 10]
# print(model)
# torch.save(model.state_dict(), 'mobilenetv1.pkl')