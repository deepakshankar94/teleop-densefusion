from segnet import SegNet as segnet
import torch 
from data_controller import SegDataset
from torch.autograd import Variable
import cv2
import numpy as np
import numpy.ma as ma
from loss import Loss
import sys
sys.path.append("..")
from lib.utils import setup_logger
import time 

model = segnet()
model = model.cuda()
model.load_state_dict(torch.load("./trained_models/model_102_0.12425611726753413.pth"))
model.eval()


def fetch_boundingboxes(rgb_image):
    npimg_ori = torch.squeeze(rgb).numpy()
    out_ori = np.transpose(npimg_ori, (1, 2, 0))
    # plt.imshow(out_ori/1200 )
    # plt.show()
    rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
    semantic = model(rgb)
    
    npimg = torch.squeeze(semantic).detach().cpu().numpy()
    out = np.transpose(npimg, (1, 2, 0))
    print(out.shape)

    crops = []
    for i in range(22):
        if i == 0:
            arr = out[:,:,i]<0
        else:
            arr = out[:,:,i]>0
        height = np.nonzero((arr).any(axis=0))[0]
        if(len(height)<30):
            continue

        height_points = [height[0],height[-1]]
        width = np.nonzero(arr.any(axis=1))[0]
        if(len(width)<30):
            continue
        width_points = [width[0],width[-1]]
        crops.append([width_points,height_points])


    return crops