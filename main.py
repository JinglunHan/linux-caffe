from src import model
from src import data
import cv2

if __name__ == '__main__':
    model_id = 1001
    device = 0
    net=model.load_model(model_id,device)

    img0 = cv2.imread('/home/roota/workstation/onnx2caffe/linux-caffe/data/dogandgirl.webp')
    img1 = data.data_pre_process(img0)

    output = model.predict(net, img1)
    
    output = data.data_post_process(output)
    img2=data.data_paint(output,img0)
    cv2.imwrite('data/result.jpg', img2)