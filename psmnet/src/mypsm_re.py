#! /usr/bin/python

from __future__ import print_function
import argparse
import glob
import os
import random
import torch
import PIL
from PIL import ImageOps
import torchvision
from torchvision.utils import save_image
from utils import preprocess 
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
import matplotlib.pyplot as plt
from matplotlib import cm
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Header
from std_msgs.msg import String
from darknet_ros_msgs.msg import BBDepth
from darknet_ros_msgs.msg import BBDepthes
from autoware_msgs.msg import DetectedObject
from autoware_msgs.msg import DetectedObjectArray
import message_filters
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import pdb

count=0
#ZED CAMERA
focal=1423.684
baseline=120.5389


#test_left_img=['./zed_left_image2.png']
#test_right_img=['./zed_right_image2.png']

#iml=cv2.imread('./left.png',1)
#imr=cv2.imread('./right.png',1)

#iml= PIL.Image.open('./left.png').convert('RGB');
#imr= PIL.Image.open('./right.png').convert('RGB')

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
# KITTI dataset pretrained model
#parser.add_argument('--loadmodel', default='/home/oyt/catkin_ws/src/psmnet/src/trained/pretrained_model_KITTI2015.tar',help='load model')

# NIA_sidewalk dataset trained model
parser.add_argument('--loadmodel', default='/home/autoware/workspace_psm/src/sidewalk_prototype_AI_Hub/psmnet/src/trained/checkpoint_3.tar',help='load model')

args = parser.parse_args()



if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')


model = nn.DataParallel(model,device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {} M'.format(np.float(sum([p.data.nelement() for p in model.parameters()]))/10**6))






def test(imgL,imgR):
    
    test_left_img=['/tmp/left.png']
    test_right_img=['/tmp/right.png']

    TestImgLoader_fix = torch.utils.data.DataLoader(
             DA.myImageFloder(test_left_img,test_right_img, False), 
             batch_size= 1, shuffle= False, num_workers= 1, drop_last=False) 
    
    batch_idx=1
    for batch_idx, (imgL, imgR) in enumerate(TestImgLoader_fix):
        model.eval()
        
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        
        # modified
        # output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]
        output = output3.data.cpu()
        
        return output

#def callback2(BB):
#       print('\nObjects : ') 
#       for box in BB.bounding_boxes:
#           print("[ %s : %6.2f %% ]"%(box.Class,box.probability*100.0))
           
           #rospy.loginfo("Xmin: {}, Xmax: {} Ymin: {}, Ymax: {}".format(box.xmin,box.xmax,box.ymin,box.ymax))
 
         
def callback(msgL,msgR,yoloBB):
       global count
       print("callback in") 
       pub = rospy.Publisher("depth", Image, queue_size=5)
       pub_bbd = rospy.Publisher("bounding_boxes_depth", BBDepthes, queue_size=5)
       pub_dds = rospy.Publisher("detection/image_detector/yolov4", DetectedObjectArray, queue_size=5)
       BBDs = BBDepthes()
       DDs = DetectedObjectArray()
       rate=rospy.Rate(10) # hz
       
       # Convert your ROS Image message to OpenCV2
       cv2_imgL = CvBridge().imgmsg_to_cv2(msgL, "bgr8")
       cv2_imgR = CvBridge().imgmsg_to_cv2(msgR, "bgr8")
       
       path = '/tmp' 
       cv2.imwrite(os.path.join(path , 'left.png'),cv2_imgL)  
       cv2.imwrite(os.path.join(path , 'right.png'),cv2_imgR)  
 
       print("\n")
       output= test(cv2_imgL, cv2_imgL)
       disp=output.numpy()
       disparity = np.transpose(disp, (1,2,0))
       disparity_scaleto_original_framesize=disparity*2  
       
       depth=(focal*baseline/disparity_scaleto_original_framesize)/1000.0
       print('Objects : ') 
       for box in yoloBB.bounding_boxes:
           xpos=int(round((box.xmin+box.xmax)/4.0)) # /2 : average , /2 :frame size 
           ypos=int(round((box.ymin+box.ymax)/4.0))
           obj_depth=depth[ypos,xpos]     
           print("[ %s : %6.4f m (%4.2f %%) ]"%(box.Class,obj_depth,box.probability*100.0))
           #
           BBD = BBDepth()
           BBD.xmin = box.xmin
           BBD.ymin = box.ymin
           BBD.xmax = box.xmax
           BBD.ymax = box.ymax
           BBD.probability = box.probability
           BBD.id = box.id
           BBD.Class = box.Class
           BBD.depth = obj_depth
           BBDs.bounding_boxes_depth.append(BBD)
           #
           DD = DetectedObject()
           DD.x = box.xmin
           DD.y = box.ymin
           DD.width = box.xmax - box.xmin
           DD.height = box.ymax - box.ymin
           DD.label = box.Class + str(obj_depth)[:3]
           DD.score = box.probability
           DD.id = box.id
           DD.user_defined_info.append(str(obj_depth)[:3])
           DDs.objects.append(DD)
           

       print("Processing frame | Delay:%6.3f" % (rospy.Time.now() - yoloBB.header.stamp).to_sec()) 
       
       #cv2.imwrite("./depth_%d.png"%(count+1),depth)
          
       print('Depth output %d' %(count+1))
       count += 1    
       #img_msg=CvBridge().cv2_to_imgmsg(depth, encoding="passthrough")
       #depth_pub = CvBridge().cv2_to_imgmsg(depth, encoding="32FC1")
       #print(np.max(depth))
       depth_clipped = np.clip(depth / 192.0, 0, 1)
       depth_colored = np.uint8(cm.jet(np.squeeze(depth_clipped)) * 255.0)
       # depth_uint8 = cv2.convertScaleAbs(depth, alpha=255.0/192.0, beta=0.0)
       depth_pub = CvBridge().cv2_to_imgmsg(depth_colored[:,:,:3], encoding="rgb8")
       pub.publish(depth_pub) 
       pub_bbd.publish(BBDs)
       pub_dds.publish(DDs)
       
       rate.sleep()  
      
       
          
          

def main():
       print("main in")
       rospy.init_node('PSM')
       
       #while not rospy.is_shutdown():
       #image_topic_l = "/zed/zed_node/left/image_rect_color"
       image_topic_l = "/kitti/camera_color_left/image_raw"
       #image_topic_r = "/zed/zed_node/right/image_rect_color"
       image_topic_r = "/kitti/camera_color_right/image_raw"
       msgL = message_filters.Subscriber(image_topic_l, Image)
       msgR = message_filters.Subscriber(image_topic_r, Image)
       yoloBB=message_filters.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes)
       # yoloBB=message_filters.Subscriber("/detection/image_detector/objects",BoundingBoxes)
       #rospy.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes,callback2)
       ts = message_filters.ApproximateTimeSynchronizer((msgL,msgR,yoloBB),5, 0.3)
       ts.registerCallback(callback)
            
       # Spin until ctrl + c
       rospy.spin()
       


if __name__ == '__main__':
   main()

