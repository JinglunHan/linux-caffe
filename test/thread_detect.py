import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import Media
from src import Model
import time
import threading


cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')
fps = cap.get(cv2.CAP_PROP_FPS)

model_id = 1001
device = 0
model=Model(model_id,device)

window_width = 640
window_height = 352
cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', window_width, window_height)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    tensor = model.predict(frame)
    img = model.paint(frame,tensor,save_image=False)
    
    cv2.imshow('Stream', img)
    # time0 =time.time()
    # cv2.imwrite( 'test_data/'+str(time0)+".png", img)