from flask import Flask,render_template,request,redirect,Response
import os
from src import Model
from src import Media
import cv2
import re
import subprocess
import threading
import time

app = Flask(__name__)
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
ffmpeg_path = '/home/roota/workstation/opencv/ffmpeg/bin/'
print(current_path,parent_directory)
record_url=[]


# 渲染 HTML 模板
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if 'task_choose' in request.form:
        task = request.form['task']
        if task == '0':
            return redirect('/media')
        elif task == '1':
            return redirect('/video')
        elif task == '2':
            return redirect('/rtsp')
    return render_template('index.html')

@app.route('/rtsp', methods=['POST','GET'])
def index_rtsp():
    if request.method == 'GET':
        return render_template('rtsp.html')
    if 'task_choose' in request.form:
        task = request.form['task']
        if task == '0':
            return redirect('/media')
        elif task == '1':
            return redirect('/video')
        elif task == '2':
            return redirect('/rtsp')
    if 'url_submit' in request.form:
        rtsp_url = request.form['rtsp_url']
        rtsp = Media(rtsp_url,1)
        if len(record_url)==0:
            record_url.append(rtsp_url)
        else:
            record_url[0]= rtsp_url
        # 检查 RTSP URL 是否有效
        # pattern = re.compile(r'rtsp://([^:]+):([^@]+)@([^:/]+)')
        # match = pattern.match(rtsp_url)
        # if not match:
        #     return render_template('rtsp.html', url_valid ='Invalid RTSP URL')
        if rtsp.media.isOpened():
            # # rtsp.write_video(ip_address='192.168.2.98')
            # write_thread = threading.Thread(target=rtsp.write_video, args=('192.168.2.98',))
            # write_thread.start()
            # time.sleep(1)
            return render_template('rtsp.html', rtsp_valid = True)
        else:
            return render_template('rtsp.html', url_valid ='Invalid RTSP URL')  
            
    return render_template('rtsp.html')

def gen(rtsp_url):
    rtsp = Media(rtsp_url,1)    
    while rtsp.media.isOpened():
        ret, frame = rtsp.media.read()
        if ret:
            ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        # 将帧作为流传输到网页
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/rtsp_show', methods=['POST','GET'])
def rtsp_show():
    rtsp_url = record_url[0]
    return Response(gen(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/media', methods=['POST','GET'])
def index_media():
    if request.method == 'GET':
        return render_template('media.html', uploaded_image=None, processed_image=None)
    if 'task_choose' in request.form:
        task = request.form['task']
        if task == '0':
            return redirect('/media')
        elif task == '1':
            return redirect('/video')
        elif task == '2':
            return redirect('/rtsp')
    file = request.files['file']
    if file.filename == '':
        return render_template('media.html', uploaded_image=None, processed_image=None)
    # Save uploaded image
    save_path = os.path.join(parent_directory+'/static/upload-img/', file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    model_id = int(request.form['model'])
    device = int(request.form['device'])
    model = Model(model_id,device)
    print(save_path)
    img0 = Media(save_path,0)
    tensor = model.predict(img0.media)
    path=model.paint(img0.media,tensor)

    uploaded_image_url = os.path.relpath(save_path, parent_directory)
    processed_image_url = os.path.relpath(path, parent_directory)
    
    return render_template('media.html', uploaded_image=uploaded_image_url, processed_image=processed_image_url)


@app.route('/video', methods=['POST','GET'])
def index_video():
    if request.method == 'GET':
        return render_template('video.html', uploaded_video=None, processed_video=None)
    if 'task_choose' in request.form:
        task = request.form['task']
        if task == '0':
            return redirect('/media')
        elif task == '1':
            return redirect('/video')
        elif task == '2':
            return redirect('/rtsp')
    file = request.files['file']
    if file.filename == '':
        return render_template('video.html', uploaded_video=None, processed_video=None)
    # Save uploaded video
    save_path = os.path.join(parent_directory+'/static/upload-video/', file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    uploaded_video_url = os.path.relpath(save_path, parent_directory)

    video0 = Media(save_path,1)
    model_id = int(request.form['model'])
    device = int(request.form['device'])
    model = Model(model_id,device)
    cap = video0.media
    codec = cv2.VideoWriter_fourcc(*'avc1')
    video_path = os.path.join(parent_directory+'/static/results_video/', 'processed_'+file.filename)
    result_path = os.path.join(parent_directory+'/static/results_video/', 'result_'+file.filename)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # out = cv2.VideoWriter(video_path, codec, 20.0, (video0.width, video0.height))
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         tensor = model.predict(frame)
    #         frame = model.paint(frame,tensor,save_image=False)
    #         out.write(frame)
    #     else:
    #         break
    
    # 定义FFmpeg命令
    ffmpeg_command = '{}ffmpeg -i {} -c:v libx264 -c:a aac -strict -2 -y {}'.format(ffmpeg_path, video_path, result_path)

    # 执行FFmpeg命令
    process = subprocess.Popen(ffmpeg_command, shell=True)
    process.wait()  # 等待命令执行完成

    # 继续执行下面的代码
    processed_video_url = os.path.relpath(result_path, parent_directory)
    return render_template('video.html', uploaded_video=uploaded_video_url, processed_video=processed_video_url)

if __name__ == '__main__':
    app.run(debug=True)