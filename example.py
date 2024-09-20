import multiprocessing.managers
from flask import Flask,render_template,request,redirect,Response
import os
from src import Model
from src import Media
import cv2
import re
import subprocess
import threading
import time
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
socketio = SocketIO(app,threaded=False)
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
ffmpeg_path = '/home/roota/workstation/opencv/ffmpeg/bin/'
print(current_path,parent_directory)
record_url=[1,1,1,1,1,1,1,1,1,1,1]
detect_img_url=[]
close_target = False

def task_choose(task):
    if task == '0':
        return '/media'
    elif task == '1':
        return '/video'
    elif task == '2':
        return '/rtsp'
    elif task == '3':
        return '/record'
    elif task == '4':
        return '/detect'
    elif task == '5':
        return '/face'
    
#region ----------------------- face ---------------------------------
db = SQLAlchemy(app)
# migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    image_path = db.Column(db.String(120), nullable=False)
    faces = db.relationship('Face', backref='user', lazy=True)
    def __repr__(self):
        return f'<User {self.username}>'

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    times = db.Column(db.Integer, nullable=True) 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    encode = db.Column(db.String,unique=True, nullable=False)
    
    def __repr__(self):
        return f'<Face {self.encodeing}>'
    
    
    
@app.before_first_request
def create_tables():
    db.create_all()

from src import FaceRecognizer
import json
import numpy as np

@app.route('/face', methods=['POST','GET'])
def index_face():
    if request.method == 'GET':
        users = User.query.all()
        faces = Face.query.all()
        return render_template('example/face.html',result = True,users = users,faces = faces )
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    if 'face_submit' in request.form:
        username = request.form['username']
        image_path = 'static/face/'+username+'.jpg'
        user_id = int(time.time())
        user = User(username=username, image_path=image_path ,id=user_id)
        db.session.add(user)
        db.session.commit()
        file = request.files['file1']
        # file.save(image_path)
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        fr = FaceRecognizer()
        face_encode_list = fr.enter_data(image,username)
        face_encode = json.dumps(face_encode_list[0].tolist())
        face = Face(id = int(time.time()),encode=face_encode,user_id = user_id)
        db.session.add(face)
        db.session.commit()
        
        users = User.query.all()
        faces = Face.query.all()
        
        return render_template('example/face.html',result = True,users = users,faces = faces)
    if 'face_detect' in request.form:
        detect_user_name_list = []
        detece_face_path_list = []
        file = request.files['file2']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        encode_list = Face.query.with_entities(Face.encode).all()
        print('encode_list',encode_list)
        for n,j in enumerate(encode_list):
            # print(n,'--------------------',type(j[0]),j[0],'---------------------------')
            encode_list[n] = json.loads(j[0])[0]
            # print(n,'--------------------',type(encode_list[n]),encode_list[n],'---------------------------')
        # print('encode_list:',encode_list)
        fd = FaceRecognizer()
        index_list,image_list = fd.compare_face_data(image,encode_list)
        print('-------index_list:',index_list)
        for n,i in enumerate(index_list):
            if i != -1:
                encode=json.dumps([encode_list[i]])
                # print('encode:',encode)
                face = Face.query.filter_by(encode=encode).first()
                user = User.query.filter_by(id=face.user_id).first()
                detect_user_name_list.append(user.username)
                path = 'static/temp/'+detect_user_name_list[n]+'.jpg'
                detece_face_path_list.append(path)
                cv2.imwrite('static/temp/'+str(detect_user_name_list[n])+'.jpg',image_list[n])
            else:
                detect_user_name_list.append('unknown'+str(n))
                path = 'static/temp/'+detect_user_name_list[n]+'.jpg'
                detece_face_path_list.append(path)
                cv2.imwrite('static/temp/'+str(detect_user_name_list[n])+'.jpg',image_list[n])
        combine_list = list(zip(detece_face_path_list,detect_user_name_list))
        return render_template('example/face.html',img_list = combine_list)

    
#endregion

#region -------------------- index ----------------------------------#
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('example/index.html')
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    
    return render_template('example/index.html')
#endregion

#region ----------------------- rtsp --------------------------------#
@app.route('/rtsp', methods=['POST','GET'])
def index_rtsp():
    if request.method == 'GET':
        return render_template('example/rtsp.html')
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    if 'url_submit' in request.form:
        rtsp_url = request.form['rtsp_url']
        rtsp = Media(rtsp_url,1)
        channel = int(request.form['channel'])
        # if len(record_url)==0:
        #     record_url.append(rtsp_url)
        # else:
        record_url[channel]= rtsp_url
        # 检查 RTSP URL 是否有效
        # pattern = re.compile(r'rtsp://([^:]+):([^@]+)@([^:/]+)')
        # match = pattern.match(rtsp_url)
        # if not match:
        #     return render_template('example/rtsp.html', url_valid ='Invalid RTSP URL')
        if rtsp.media.isOpened():
            # # rtsp.write_video(ip_address='192.168.2.98')
            # write_thread = threading.Thread(target=rtsp.write_video, args=('192.168.2.98',))
            # write_thread.start()
            # time.sleep(1)
            return render_template('example/rtsp.html', rtsp_valid = True)
        else:
            return render_template('example/rtsp.html', url_valid ='Invalid RTSP URL')  
    if 'channel_close' in request.form:
        close_target = True
        for i in range(len(record_url)):
            if i == 1:
                pass
            else:
                record_url[i] = 1
        time.sleep(1)
        close_target = False
    return render_template('example/rtsp.html')

def gen(rtsp_url,detect=False):
    global close_target
    rtsp = Media(rtsp_url,1)    
    while rtsp.media.isOpened():
        ret, frame = rtsp.media.read()
        if ret:
            ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        if close_target:
            break
        # 将帧作为流传输到网页
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/rtsp_show/<int:channel>', methods=['POST','GET'])
def rtsp_show(channel):
    rtsp_url = record_url[channel] 
    if rtsp_url != 1:
        return Response(gen(rtsp_url,detect=False), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        pass

import multiprocessing
process_url = multiprocessing.Manager().list()
reload_model = multiprocessing.Manager().Value('i',0)

@app.route('/rtsp_detect', methods=['POST','GET'])
def rtsp_detect():
    rtsp_url = record_url[1]
    if 'model_choose' in request.form:
        print('------------------ rtsp detect -------------------------')
        time.sleep(1)
        model_id = int(request.form['model'])
        device = int(request.form['device'])
        detect_process = multiprocessing.Process(target=rtsp_detect_thread, args=(model_id,device,rtsp_url,reload_model))
        reload_model.value = 0
        detect_process.start()
    if 'model_reload' in request.form:
        reload_model.value = 1
        time.sleep(1)
        reload_model.value = 0
        # detect_thread = threading.Thread(target=rtsp_detect_thread, args=(model_id,device,rtsp_url))
        # reload_model = False
        # detect_thread.start()
    return render_template('example/rtsp.html', rtsp_valid = True,detect_rtsp=True)
def rtsp_detect_thread(model_id,device,rtsp_url,reload_model):
    model = Model(model_id,device)
    rtsp = Media(rtsp_url,1) 
    step = 5
    count = 0   
    while rtsp.media.isOpened() :
        if reload_model.value == 1:
            break
        ret, frame = rtsp.media.read()
        count += 1
        if ret and count==step:
            count = 0
            tensor=model.predict(frame)
            path = model.paint(frame,tensor,save_image=True)
            if path != None:
                # print(path)
                path =  os.path.relpath(path, parent_directory)
                # print(path)
                # if len(detect_img_url)==0:
                #     detect_img_url.append(path)
                # else:
                #     detect_img_url[0] = path
                if len(process_url)==0:
                    process_url.append(path)
                else:
                    process_url[0] = path

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(send_result)
def send_result():
    # url = detect_img_url
    url = process_url
    while True:
        if len(url)>0:
            socketio.emit('detect_img', url[0])
        time.sleep(1)
        # url = detect_img_url

    # socketio.start_background_task()

#endregion

#region -------------------- media html ------------------------------#
@app.route('/media', methods=['POST','GET'])
def index_media():
    if request.method == 'GET':
        return render_template('example/media.html', uploaded_image=None, processed_image=None)
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    file = request.files['file']
    if file.filename == '':
        return render_template('example/media.html', uploaded_image=None, processed_image=None)
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
    
    return render_template('example/media.html', uploaded_image=uploaded_image_url, processed_image=processed_image_url)
#endregion

#region -------------------- video html ------------------------------#
@app.route('/video', methods=['POST','GET'])
def index_video():
    if request.method == 'GET':
        return render_template('example/video.html', uploaded_video=None, processed_video=None)
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    file = request.files['file']
    if file.filename == '':
        return render_template('example/video.html', uploaded_video=None, processed_video=None)
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
    return render_template('example/video.html', uploaded_video=uploaded_video_url, processed_video=processed_video_url)

#endregion

#region -------------------- record html ------------------------------#
@app.route('/record', methods=['POST','GET'])
def index_record():
    directory = get_directory_structure(parent_directory+'/static/record')
    # print(directory)
    if request.method == 'GET':
        return render_template('example/record.html', directory=directory)
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    if 'url_submit' in request.form:
        path = request.form['record_file_url']
        path = 'static/record/'+path
        print(path)
        return render_template('example/record.html', directory=directory, record_video=path)

def get_directory_structure(rootdir):
    """
    获取目录结构的函数
    """
    directory_structure = {}
    for root, dirs, files in os.walk(rootdir):
        dirs.sort()
        files.sort()
        for dir in dirs:
            directory_structure[dir] = os.listdir(os.path.join(root, dir))
    return directory_structure

#endregion

#region -------------------- detect html ------------------------------#
from datetime import datetime
@app.route('/detect', methods=['POST','GET'])
def index_detect():
    if request.method == 'GET':
        return render_template('example/detect.html')
    if 'task_choose' in request.form:
        task = request.form['task']
        target = task_choose(task)
        return redirect(target)
    
    if 'model_choose' in request.form:
        modelkind = request.form['model']
        result_path = parent_directory + '/static/results/' + modelkind
        data = request.form['selected_date']
        print(result_path,data)
        ymd_data_list = []
        detect_url_list = []
        try:
            files = os.listdir(result_path)
        except:
            return render_template('example/detect.html',result = False)
        files.sort()
        for file in files:      
            date_time_str = file.split('.')[0]
            ymd_data_time = date_time_str.split(' ')[0]
            detect_url_list.append('static/results/'+modelkind+'/'+file)
            if ymd_data_time not in ymd_data_list:
                ymd_data_list.append(ymd_data_time)
        
        if data not in ymd_data_list:
            return render_template('example/detect.html',result = True,ymd_data_list=ymd_data_list)
        else:
            return render_template('example/detect.html',result = True,ymd_data_list=ymd_data_list,detect_url_list=detect_url_list)
        

#endregion

if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app,debug=True,host='0.0.0.0',port=5000)