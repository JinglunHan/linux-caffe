import caffe
import numpy as np
import os
import logging
import time
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
abs_path = os.path.dirname(parent_directory)
import sys
sys.path.append(abs_path + '/src/')
import src.data_op as dop


# logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ID ={
    1001:'x8_yolov8m-pose',

    1101:'smokefire',
    1102:'fire',

    1201:'helmet',

    1301:'beltsprinklingcoal'
}


class Model():
    def __init__(self, model_id,device):
        self.id = model_id
        self.device = device            # device: 0=cpu,1=gpu
        self.net = load_model(model_id,self.device)
        self.task = model_task(model_id)
    
    def predict(self,image):
        start = time.time()
        image = dop.data_pre_process(image,self.task)
        task = self.task
        net = self.net
        if task == 1:
            net.blobs['images'].data[...] = image.tolist()
            net.forward()
            output = net.blobs['output'].data
            end = time.time()
            logging.info("Model prediction in {} seconds".format(end-start))
            return output
        elif task == 0:
            net.blobs['_Input1'].data[...] = image.tolist()
            net.forward()
            output1 = net.blobs['25_Detect1_Conv2d1'].data
            output2 = net.blobs['25_Detect1_Conv2d2'].data
            output3 = net.blobs['25_Detect1_Conv2d3'].data
            end = time.time()
            logging.info("Model prediction in {} seconds".format(end-start))
            return [output1,output2,output3]
    
    def paint(self,img,tensor, 
              box_line=3,
              paint_box=True,
              pose_kind=0,
              line_color=(0,0,255),
              line_weight=3,
              on_original_img=True,
              save_image=True,
              save_path=abs_path+'/static/results/'):
        if self.task == 0:
            tensor = dop.data_detect_process(tensor)
            if tensor[0][0] != -1:
                img = dop.detect_paint(tensor,img)
            else:
                return None

        elif self.task == 1:
            tensor = dop.data_post_process(tensor)
            if tensor[0][0][0] != -1:
                img = dop.data_paint(tensor,img)
            else:
                return None
        
        if save_image == True:
            from datetime import datetime
            import cv2
            name = MODEL_ID[self.id]
            time = datetime.now()
            target_dir = save_path+name
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            path = target_dir+'/'+str(time)+'.jpg'
            cv2.imwrite(path,img)   
            return  path
        return img

            

    
def load_model(model_id,device):
    start = time.time()
    if model_id not in MODEL_ID:
        logging.error("Model id not found")
        return None
    
    model_name = MODEL_ID[model_id]
    prototxt_path = abs_path+"/models/"+model_name+".prototxt"
    caffemodel_path = abs_path+"/models/"+model_name+".caffemodel"
    
    if prototxt_path is None or caffemodel_path is None:
        logging.error("Model path not found")
        return None
    
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    if device == 1:
        caffe.set_mode_gpu()
    end = time.time()
    logging.info("Model loaded in {} seconds".format(end-start))
    return net

def model_task(model_id):
    yolov5_detect = [1101,1102,1201,1301]
    yolov8_pose = [1001]
    if model_id in yolov5_detect:
        return 0
    elif model_id in yolov8_pose:
        return 1
    
