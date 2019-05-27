from segnet import SegNet as segnet
import torch 
from data_controller import SegDataset
from torch.autograd import Variable
import cv2
import numpy as np
import torchvision.transforms as transforms
import numpy.ma as ma
from loss import Loss
import matplotlib.pyplot as plt
import sys
import time 

model = segnet()
model = model.cuda()
import os
dirname = os.path.dirname(__file__)
model.load_state_dict(torch.load(os.path.join(dirname, "trained_models/model_102_0.12425611726753413.pth")))
model.eval()


def fetch_boundingboxes(rgb):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #rgb = np.array(rgb_image.convert("RGB"))
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    rgb = rgb.unsqueeze(0)
    rgb = Variable(rgb).cuda()
    semantic = model(rgb)
    
    npimg = torch.squeeze(semantic).detach().cpu().numpy()
    out = np.transpose(npimg, (1, 2, 0))
    #print(out.shape)

    crops = []
    for i in range(22):
        if i == 0:
            continue #ignore the crop with all the objects
        else:
            arr = out[:,:,i]>0
        height = np.nonzero((arr).any(axis=0))[0]
        if(len(height)<100):
            continue

        height_points = [height[0],height[-1]]
        width = np.nonzero(arr.any(axis=1))[0]
        if(len(width)<30):
            continue
        width_points = [width[0],width[-1]]
        if i == 20:  #only for meat can
            crops.append([width_points,height_points])


    return np.array(crops)