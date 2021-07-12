# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:07:21 2021

@author: CHEN
"""
"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, my_load_single_img
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def run(img_np = None,
        weights='yolov5s.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        half=False,  # use FP16 half-precision inference
        ):
    
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    img = my_load_single_img(img_np, img_size=imgsz, stride=stride)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    if len(pred[0]):
        # Rescale boxes from img_size to im0 size
        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img_np.shape).round()
    
    return pred[0].cpu().numpy()
    

if __name__ == "__main__":

    check_requirements(exclude=('tensorboard', 'thop'))
    
    key_image = np.array(cv2.imread('data/images/zidane.jpg'))
    
    key_detresult = run(img_np = key_image,
                        weights = 'pre_weights/yolov5x6.pt',
                        classes = 0,
                        conf_thres = 0.5)
    
    for i in range(len(key_detresult)):
        label = 'person' + f'{key_detresult[i][4]:.2f}'
        c1 = (int(key_detresult[i][0]), int(key_detresult[i][1]))
        c2 = (int(key_detresult[i][2]), int(key_detresult[i][3]))
        cv2.rectangle(key_image, c1, c2, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness = 2)[0]
        c3 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(key_image, c1, c3, (0, 0, 255), -1, cv2.LINE_AA)  # filled
        cv2.putText(key_image, label, (c1[0], c1[1] - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        
    cv2.imshow("mydet", key_image)
