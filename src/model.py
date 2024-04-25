import caffe
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

#model name and it's id
_MODEL_ID ={
    1001:'x8_yolov8m-pose'
}

# get model path
def get_model_path(model_id):
    if model_id not in _MODEL_ID:
        logging.error("Model id not found")
        return None
    
    model_name = _MODEL_ID[model_id]
    prototxt_path = "./models/"+model_name+".prototxt"
    caffemodel_path = "./models/"+model_name+".caffemodel"
    return prototxt_path, caffemodel_path

    
def load_model(model_id,device):
    prototxt_path, caffemodel_path = get_model_path(model_id)
    if prototxt_path is None or caffemodel_path is None:
        logging.error("Model path not found")
        return None
    
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    if device == 'gpu' or device == 1:
        net.set_mode_gpu()

    return net

def predict(net, image):
    net.blobs['images'].data[...] = image.tolist()
    net.forward()
    output = net.blobs['output'].data
    return output