#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import operator

def detect(net, im, objlist):
    """Detect object classes in an image using pre-computed object proposals."""
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    scores, boxes = im_detect(net, im)
    bboxlist = []
    scorelist = []
    for i in objlist:
        CONF_THRESH = 0.0
        NMS_THRESH = 0.0

        cls_ind = i
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)

        bbox = dets[0, :4]

        cls_scores_sorted = sorted(cls_scores)
        
        sort_index = np.argsort(cls_scores)
        bbox = (cls_boxes[sort_index[-1], :4])

        bboxlist.append(bbox)
        scorelist.append(cls_scores_sorted[-1])
    return bboxlist, scorelist

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args