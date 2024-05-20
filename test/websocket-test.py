
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime
import time
import os

app = Flask(__name__)
socketio = SocketIO(app)
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
print(parent_directory)

@app.route('/')
def index():
    return render_template('websocket.html')

def update_time():
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        socketio.emit('update_time', current_time)
        time.sleep(1)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('update_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    socketio.start_background_task(handle_image)

# @socketio.on('image')
def handle_image():
    # print('image connected')
    folder_path = '/home/roota/workstation/onnx2caffe/linux-caffe/test/static/upload-img'
    image_files = list_image_files(folder_path)
    while True:
        for image_file in image_files:
            print(image_file)
            image_url = os.path.relpath(image_file, parent_directory)
            print(image_url)
            socketio.emit('image', image_url)
            time.sleep(1)  # 等待1秒


def list_image_files(folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图片文件扩展名
    image_files = []

    # 遍历目标文件夹
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 检查文件扩展名是否在图片扩展名列表中
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # 如果是图片文件，将其路径添加到列表中
                image_files.append(os.path.join(root, file))

    return image_files


if __name__ == '__main__':
    socketio.start_background_task(update_time)
    socketio.run(app,debug=True)