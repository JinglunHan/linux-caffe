from flask import Flask, Response,render_template
import cv2
import sys
sys.path.append('/workdir_test/0_hjl/git/linux-caffe')
from src import model
from src import data
import cv2
import time

app = Flask(__name__)

def gen_frames(ai=True):
    # 使用 OpenCV 捕获 RTSP 流
    cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    if ai == True:
        model_id = 1001
        device = 0
        net=model.load_model(model_id,device)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        else:
            if ai == True:
                img = data.data_pre_process(frame)
                output = model.predict(net, img)
                output = data.data_post_process(output)
                frame = data.data_paint(output, frame,on_original_img=False)
            # 将捕获到的帧编码为 JPEG 格式
                ret, buffer = cv2.imencode('.png', frame)
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret:
                break

            # 将帧作为流传输到网页
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # 返回网页模板，包含视频播放区域
    return render_template('double-img.html')

@app.route('/video_feed')
def video_feed():
    # 返回生成的帧流
    return Response(gen_frames(ai=False), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames(ai=False), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)