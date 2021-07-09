# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions
from multiprocessing import Queue

sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_detection_dataset import DC # dataset config
import sunrgbd_utils


#----- For coral EdgeTPU inference
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter


def preprocess_point_cloud(point_cloud, mask_expand=None, num_class=0, calib_filepath=None):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now    

    if not mask_expand is None:
        h, w, _ = mask_expand.shape

        calib = sunrgbd_utils.SUNRGBD_Calibration(calib_filepath)
        uv,d = calib.project_upright_depth_to_image(point_cloud[:,0:3])     
        
        pc_shape = point_cloud.shape #(N, 3 + features)
        painted_shape = list(pc_shape).extend([DC.num_class]) #(N, 3 + features + image seg classification)
        painted = np.zeros(painted_shape)                

        # Find the mapping from image to point cloud        
        uv[:,0] = np.minimum(w-1, uv[:,0]) # Width
        uv[:,1] = np.minimum(h-1, uv[:,1]) # Height

        # Append segmentation score to each point
        # mask_expand has (h, w) format, wherease uv has (w, h) format
        point_cloud = np.concatenate([point_cloud,\
                                mask_expand[uv[:,1].astype(np.int), uv[:,0].astype(np.int)]], axis=-1) 

    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)

    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def votenet_inference(queue):

    use_painted = True
    threshold = 0.9
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    checkpoint_path = os.path.join(demo_dir, 'checkpoint_210607.tar')
    calib_filepath = os.path.join(demo_dir,'calib_rs.txt')
    #COCO2SUNRGBD = {62:3, 63:2, 65:0, 67:1, 70:4, 1:10}
    VOC2SUNRGBD = dict()
    for i in range(21):
        VOC2SUNRGBD[i] = 0
    VOC2SUNRGBD[15] = 10 # person
    VOC2SUNRGBD[9] = 3 #chair
    VOC2SUNRGBD[11] = 1 #table
    VOC2SUNRGBD[18] = 2 #sofa

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = 1 + DC.num_class if use_painted else 1
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=input_dim, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
   
    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)

    if use_painted:
        #Load model
        imgSegModel = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        imgSegModel.eval()    
        imgSegModel.to(device)
    
    filename = queue.get()
    pc_dir = os.path.join(os.path.dirname(BASE_DIR), 'point_cloud')
    pc_path = os.path.join(pc_dir, filename)
    point_cloud = read_ply(pc_path)

    image_dir = os.path.join(os.path.dirname(BASE_DIR), 'rgb_image')
    image_path = os.path.join(image_dir, filename[:-6] + '.jpg')

    # Iamge segmentation
    if use_painted:
        interpreter = make_interpreter('deeplabv3_mnv2_pascal_quant_edgetpu.tflite', device=':0')
        interpreter.allocate_tensors()
        width, height = common.input_size(interpreter)

        img = Image.open(args.input)

        #For now, let's keep aspect ratio
        keep_aspect_ratio = True
        if keep_aspect_ratio:
            resized_img, _ = common.set_resized_input(
                interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
        else:
            resized_img = img.resize((width, height), Image.ANTIALIAS)
            common.set_input(interpreter, resized_img)
        interpreter.invoke()
        result = segment.get_output(interpreter)

        if len(result.shape) == 3:
            result = np.argmax(result, axis=-1)

        new_width, new_height = resized_img.size
        result = result[:new_height, :new_width] #(h, w) format, each pixel's value is label number

        # Convert back to original size
        resized_result = result.resize(img.size)

        # Class number conversion
        v = np.array(list(VOC2SUNRGBD.values()))
        k = np.array(list(VOC2SUNRGBD.keys()))
        for i in range(resized_result.shape[1]):
            _, C =  np.where(resized_result[:,i][:,None] == k)
            resized_result[:,i] = v[C]

        result_mask = resized_result > 0

        #To one hot vector        
        mask_expand = np.tile(np.expand_dims(result_mask, axis=-1), (1,1,DC.num_class)) \
                    * np.eye(DC.num_class)[resized_result]

        '''
        #Load image
        img = cv2.imread(image_path)       
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).to(device) 

        result = segment.get_output(interpreter)

        # Img segmentation model inference
        pred = imgSegModel([img_tensor])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())        
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        if len(masks.shape) < 3:
            masks = np.expand_dims(masks, axis = 0)
        pred_score_thr = [pred_score.index(x) for x in pred_score if x > threshold]
        mask_expand = np.zeros((masks.shape[1], masks.shape[2], DC.num_class)) # h, w, num_class 

        if len(pred_score_thr) > 0:        
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]            
            masks = masks[:(pred_t+1)] # (prediction, h, w)
            
            # Segmentation score to each pixel in one hot form
            for i in range(pred_t+1):
                each_class = pred[0]['labels'][i].item()                
                if each_class not in VOC2SUNRGBD:
                    #print("Not found", each_class)
                    continue
                print(each_class)
                new_class = VOC2SUNRGBD[each_class]               
                mask_expand += np.tile(np.expand_dims(masks[i] * pred_score[i], axis=-1), (1,1,DC.num_class)) \
                                * np.eye(DC.num_class)[new_class]  # (h, w, num_class)          
        '''
        pc = preprocess_point_cloud(point_cloud, mask_expand, DC.num_class, calib_filepath)
    else:
        pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))
   
    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    #print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))
    
    ## Added to print objects

    TYPE2CLASS_DICT = {0:'bed', 1:'table', 2:'sofa', 3:'chair', 4:'toilet', 5:'desk', 6:'dresser', 7:'night_stand', 8:'bookshelf', 9:'bathtub', 10:'person'}

    for pred in pred_map_cls[0]:
        print("Class:", TYPE2CLASS_DICT[pred[0]])
        print("Score:", pred[2])
        print("Heading angle:", pred[3])
        print("Coordinates:", pred[1])
        

    ## End of added code
  
    #dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    #if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    #MODEL.dump_results(end_points, dump_dir, DC, True)
    #print('Dumped detection results to folder %s'%(dump_dir))

    queue.put(pred_map_cls)
