import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import model
from src import data
import cv2
import time
import threading

ai =True
cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
if ai == True:
    model_id = 1001
    device = 0
    net=model.load_model(model_id,device)

# window_width = 640
# window_height = 352
# cv2.namedWindow('Network Stream', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Network Stream', window_width, window_height)

def predict_and_append_result(net, img, result_list):
    output = model.predict(net, img)
    result_list.append(output)

def pre_process_and_append_img(frame, img_list):
    img = data.data_pre_process(frame,task=1)
    img_list.append(img)

def post_and_paint_append_jpg(result, frame,jpg_list):
    output = data.data_post_process(result)
    frame = data.data_paint(output, frame)
    jpg_list.append(frame)

def show():
    window_width = 640
    window_height = 352
    cv2.namedWindow('Network Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Network Stream', window_width, window_height)
    while True:
        if jpg_list:
            cv2.imshow('Network Stream', jpg_list[0])
            # time0 =time.time()
            # cv2.imwrite( 'test_data/'+str(time0)+".png", jpg_list[0])
            del jpg_list[0]
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            

# 创建一个空列表，用于存储结果
img_list = []
result_list = []
jpg_list = []
count,step = 0,5

show_thread = threading.Thread(target=show)
show_thread.start()

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    if count == step:
        pre_process_and_append_img(frame, img_list)
        count = 0
    else:
        count += 1


    if img_list:
        predict_thread = threading.Thread(target=predict_and_append_result, args=(net, img_list[0], result_list))
        del img_list[0]
        predict_thread.start()
    
    if result_list:
        paint_thread = threading.Thread(target=post_and_paint_append_jpg, args=(result_list[0], frame, jpg_list))
        del result_list[0]
        paint_thread.start()



        

# while cap.isOpened():
#     time0 =time.time()
#     ret, frame = cap.read()
#     time1 = time.time()
#     print('---------- frame read time : ',time1-time0)
#     if not ret:
#         break
#     else:
#         if ai == True:
#             timea = time.time()
#             img = data.data_pre_process(frame,task=1)
#             timeb = time.time()
#             output = model.predict(net, img)
#             timec = time.time()
#             output = data.data_post_process(output)
#             timed = time.time()
#             frame = data.data_paint(output, frame)
#             timed = time.time()
#             print(' img preprocess time: ',timeb-timea,' predict time: ',timec-timeb)
#             print(' post process time: ',timed-timec,' paint time: ',timed-timec)
#         # 将捕获到的帧编码为 JPEG 格式
#         #     ret, buffer = cv2.imencode('.png', frame)
#         # else:
#         #     ret, buffer = cv2.imencode('.jpg', frame)
#             frame = cv2.resize(frame, (window_width, window_height))
#         else:
            
#             frame = cv2.resize(frame, (window_width, window_height))
#             time3 = time.time()
#             print('________________ frame end time :',time3-time0)

#         # 在窗口中显示帧
#         cv2.imshow('Network Stream', frame)

#         # 按下 'q' 键退出循环
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break