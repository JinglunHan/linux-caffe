from flask import Flask, Response,render_template
import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import Model
from src import Media
import cv2
import time
import threading

app = Flask(__name__)
model_id = 1001
device = 0
model=Model(model_id,device)
tensor_list = []
img_list = []
lock = threading.Lock()
lock1 = threading.Lock()

def tensor_append(frame):
    output = model.predict(frame)
    lock.acquire()
    tensor_list.append(output)
    lock.release()

def img_paint_append(frame,tensor):
    lock.acquire()
    frame = model.paint(frame,tensor,save_image=False)
    lock.release()
    lock1.acquire()
    img_list.append(frame)
    lock1.release()

def frames_show():
    print('start frames_show')
    while True:
        if img_list:
            lock1.acquire()
            buffer = cv2.imencode('.jpg', img_list[0])
            del img_list[0]
            lock1.release()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
# show_thread = threading.Thread(target=frames_show)
# show_thread.start()

    
def gen_frames(ai=True):
    # 使用 OpenCV 捕获 RTSP 流
    cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
        
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        else:
            if ai==True:
                predict_thread = threading.Thread(target=tensor_append,args=(frame,))
                predict_thread.start()

                if tensor_list:
                    paint_thread = threading.Thread(target=img_paint_append,args=(frame,tensor_list[0]))
                    paint_thread.start()
                    lock.acquire()
                    del tensor_list[0]
                    lock.release()
                if img_list:
                    lock1.acquire()
                    ret, buffer = cv2.imencode('.jpg', img_list[0])
                    del img_list[0]
                    lock1.release()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            # if ai == True:
            #     output = model.predict(frame)
            #     frame = model.paint(frame,output,save_image=False)
            # # 将捕获到的帧编码为 JPEG 格式
            #     ret, buffer = cv2.imencode('.png', frame)
            # else:
            #     ret, buffer = cv2.imencode('.jpg', frame)
            
            # if not ret:
            #     break

            # # 将帧作为流传输到网页
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # cap.release()

@app.route('/')
def index():
    # 返回网页模板，包含视频播放区域
    return render_template('double-img.html')

@app.route('/video_feed')
def video_feed():
    # 返回生成的帧流
    # print('---------- detect -------------')
    # show_thread = threading.Thread(target=gen_frames,args=(True,))
    # show_thread.start()
    # print('start show_thread')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(ai=False), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True,threaded=False,processes=3)