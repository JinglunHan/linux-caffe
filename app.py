
from flask import Flask, render_template, Response
import cv2
# import sys
# sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import model
from src import data
import cv2
import base64
import time

app = Flask(__name__)

Rtsp_id = {
    1:'rtsp://admin:admin12345@192.168.6.64:554/Streaming/Channels/103?transportmode=unicast&profile=Profile_3',
    2:'rtsp://admin:admin12345@192.168.6.196:554/Streaming/Channels/102?transportmode=unicast&profile=Profile_2',
    3:'rtsp://192.168.2.198:8554/micagent1',
    4:'rtsp://admin:admin12345@192.168.2.224:554/mpeg4cif',
    5:'rtsp://admin:admin12345@192.168.6.199:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif',
    6:'rtsp://192.168.2.198:8554/micagent2',
    7:'rtsp://192.168.2.198:8554/micagent3'
}

def gen_frames(rtsp_id):
    # 使用 OpenCV 捕获 RTSP 流
    # cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')
    cap = cv2.VideoCapture(Rtsp_id[rtsp_id])
    # cap = cv2.VideoCapture('data/move.mp4')
    model_id = 1001
    device = 1
    net=model.load_model(model_id,device)
    
    frame_gap = 5
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if frame_count == frame_gap:
            img = data.data_pre_process(frame)
            # if len(img) == 1:
            #     print(len(img))
            #     continue
            output = model.predict(net, img)
            output = data.data_post_process(output)
            if output !=0:
                frame = data.data_paint(output, frame,pose_kind=1)

            if not ret:
                break
            else:
                # 将捕获到的帧编码为 JPEG 格式
                ret, buffer = cv2.imencode('.jpg', frame)
                
                if not ret:
                    break

                # 将帧作为流传输到网页
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            frame_count = 0

    cap.release()

@app.route('/')
def index():
    # 返回网页模板，包含视频播放区域
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 返回生成的帧流
    return Response(gen_frames(rtsp_id=1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    # 返回生成的帧流
    return Response(gen_frames(rtsp_id=2), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)