import caffe
import logging
import time
import os
import numpy as np

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
abs_path = os.path.dirname(parent_directory)

#model name and it's id
_MODEL_ID ={
    1001:'x8_yolov8m-pose',
    1101:'smokefire',
    1102:'fire',
    1201:'helmet',
    1301:'beltsprinklingcoal'
}

# get model path
def get_model_path(model_id):
    if model_id not in _MODEL_ID:
        logging.error("Model id not found")
        return None
    
    model_name = _MODEL_ID[model_id]
    prototxt_path = abs_path+"/models/"+model_name+".prototxt"
    caffemodel_path = abs_path+"/models/"+model_name+".caffemodel"
    return prototxt_path, caffemodel_path

    
def load_model(model_id,device):
    start = time.time()
    prototxt_path, caffemodel_path = get_model_path(model_id)
    if prototxt_path is None or caffemodel_path is None:
        logging.error("Model path not found")
        return None
    
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    if device == 'gpu' or device == 1:
        caffe.set_mode_gpu()
    end = time.time()
    print('load model : ',end-start)
    return net


num0 = 0

def predict(net, image,task=0):
    # caffe.set_mode_gpu()
    start = time.time()
    if task == 0:
        net.blobs['images'].data[...] = image.tolist()
        net.forward()
        output = net.blobs['output'].data
        return output
    elif task == 1:
        net.blobs['_Input1'].data[...] = image.tolist()
        net.forward()
        output1 = net.blobs['25_Detect1_Conv2d1'].data
        output2 = net.blobs['25_Detect1_Conv2d2'].data
        output3 = net.blobs['25_Detect1_Conv2d3'].data
        # nc = int(output1.shape[1]/3-5)

        # output1 = output1.reshape((1,(nc+5)*3,-1))
        # output2 = output2.reshape((1,(nc+5)*3,-1))
        # output3 = output3.reshape((1,(nc+5)*3,-1))
        # output = np.concatenate((output1,output2,output3),axis=2)
        # output1 = output[:,:nc+5,:]
        # output2 = output[:,nc+5:(nc+5)*2,:]
        # output3 = output[:,(nc+5)*2:(nc+5)*3,:]
        # output = np.concatenate((output1,output2,output3),axis=2)
        return [output1,output2,output3]

    
        

    