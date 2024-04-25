import cv2

def pull_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        # 在这里可以对帧进行处理，例如显示到窗口或保存到文件
        cv2.imshow('RTSP Stream', frame)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# # 拉取单个 RTSP 流的示例
# rtsp_url = "rtsp://192.168.2.198:8554/micagent1"
# pull_rtsp_stream(rtsp_url)

from flask import Flask, Response
import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import model
from src import data
import cv2
import time

app = Flask(__name__)

def gen_frames():
    # 使用 OpenCV 捕获 RTSP 流
    cap = cv2.VideoCapture('rtsp://192.168.2.198:8554/micagent1')

    model_id = 1001
    device = 0
    start_time = time.time()
    net=model.load_model(model_id,device)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        else:
            # 将捕获到的帧编码为 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            img = data.data_pre_process(buffer)
            output = model.predict(net, img)
            output = data.data_post_process(output)
            img = data.data_paint(output, buffer)
            if not ret:
                break

            # 将帧作为流传输到网页
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # 返回网页模板，包含视频播放区域
    return '''
    <html>
    <head>
    <title>RTSP Stream</title>
    </head>
    <body>
    <h1>RTSP Stream</h1>
    <img src="/video_feed">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    # 返回生成的帧流
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)