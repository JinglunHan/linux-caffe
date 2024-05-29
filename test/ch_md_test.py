import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import Model
from src import Media
import cv2
import time
import threading
import multiprocessing

url = 'static/move.mp4'
output_path = 'static/'
modelid = 1001
device = 0
cap = Media(url,1)
#multithread
tensor_list = []
img_list = []
lock = threading.Lock()
lock1 = threading.Lock()
done_target = False
tensor_dict = {}
img_dict = {}

# region one model and one thread
def one_and_one():
    time_start = time.time()
    out = cv2.VideoWriter(output_path+'one_and_one.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    model = Model(modelid, device)
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        if not ret:
            break
        tensor = model.predict(frame)
        img = model.paint(frame,tensor,save_image=False)
        out.write(img)
    time_end = time.time()
    print('one_and_one takes time : ', time_end - time_start) 
# endregion

# region  one model and multithread
def one_and_multi():
    global img_list,tensor_list,done_target
    time_start = time.time()
    model = Model(modelid, device)
    out = cv2.VideoWriter(output_path+'one_and_multi.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    paint_thread = threading.Thread(target=oam_paint, args=(model,out))
    paint_thread.start()
    count = 0
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        if not ret:
            break
        with lock1:
            img_list.append(frame)
        count += 1
        print('count : ',count)
        tensor_thread = threading.Thread(target=oam_predict, args=(frame,model,count))
        tensor_thread.start()
    done_target = True
    paint_thread.join()
    time_end = time.time()
    print('one_and_multi takes time : ', time_end - time_start)

def oam_predict(img,model,count):
    global img_list,tensor_list
    frame = img
    tensor = model.predict(frame)
    print('oam_predict: ',count)
    with lock:
        tensor_list.append(tensor)
   
def oam_paint(model,out):
    global img_list,tensor_list,done_target
    num =0
    print('num :',num)
    while True:
        if tensor_list and img_list:
            num +=1
            print('num :',num)
            with lock1:
                img = img_list[0]
            with lock:
                tensor = tensor_list[0]

            img = model.paint(img,tensor,save_image=False)
            with lock1:
                del img_list[0]
            with lock:
                del tensor_list[0]
            out.write(img)
        elif done_target == True:
            break
# endregion

# region one model and multithread in dict
def one_and_multi_dict():
    global tensor_dict,img_dict,done_target
    time_start = time.time()
    model = Model(modelid, device)
    out = cv2.VideoWriter(output_path+'one_and_multi_dict.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    paint_thread = threading.Thread(target=oamd_paint, args=(model,out))
    paint_thread.start()
    count = 0
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        if not ret:
            break
        with lock1:
            img_dict[count] = frame
        tensor_thread = threading.Thread(target=oamd_predict, args=(frame,model,count))
        tensor_thread.start()
        count += 1
        print('count : ',count)
    done_target = True
    paint_thread.join()
    time_end = time.time()
    print('one_and_multi_dict takes time : ', time_end - time_start)

def oamd_predict(frame,model,count):
    global tensor_dict,img_dict
    tensor = model.predict(frame)
    print('oamd predict : ',count)
    with lock:
        tensor_dict[count] = tensor

def oamd_paint(model,out):
    global tensor_dict,img_dict,done_target
    num = 0
    while True:
        if num in img_dict and num in tensor_dict:
            with lock1:
                img = img_dict[num]
            with lock:
                tensor = tensor_dict[num]
            img = model.paint(img,tensor,save_image=False)
            out.write(img)
            with lock1:
                del img_dict[num]
            with lock:
                del tensor_dict[num]
            print('num :',num)
            num += 1
        elif done_target:
            break
            
# endregion

# region two model 
def two_and_mul():
    global tensor_dict,img_dict,done_target
    time_start = time.time()
    out = cv2.VideoWriter(output_path+'two_and_mul.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    model = Model(modelid, device)
    model1 = Model(modelid, device)
    paint_thread = threading.Thread(target=oamd_paint, args=(model,out))
    paint_thread.start()
    count = 0
    step = 0
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        if not ret:
            break
        with lock1:
            img_dict[count] = frame
        if step == 1:
            tensor_thread = threading.Thread(target=oamd_predict, args=(frame,model1,count))
        else:
            tensor_thread = threading.Thread(target=oamd_predict, args=(frame,model,count))
        tensor_thread.start()
        count += 1
        print('count : ',count)
    done_target = True
    paint_thread.join()
    time_end = time.time()
    print('two_and_multi_dict takes time : ', time_end - time_start)

# endregion

# region two process,one for predict and one for paint 
process_tensor_dict = multiprocessing.Manager().dict()
process_img_dict = multiprocessing.Manager().dict()
process_done_target = multiprocessing.Manager().Value('i',0)
lock3 = multiprocessing.Lock()
lock4 = multiprocessing.Lock()

def one_and_mulprocess_dict():
    global tensor_dict,img_dict,done_target
    time_start = time.time()
    model = Model(modelid, device)
    out = cv2.VideoWriter(output_path+'one_and_mulprocess_dict.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    paint_thread = threading.Thread(target=oamd_paint, args=(model,out))
    paint_thread.start()
    count = 0
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        if not ret:
            break
        with lock3:
            process_img_dict[count] = frame
        tensor_process = multiprocessing.Process(target=oamd_pro_predict, args=(frame,model,count))
        tensor_process.start()
        count += 1
        print('count : ',count)
    done_target = True
    paint_thread.join()
    time_end = time.time()
    print('one_and_multi_dict takes time : ', time_end - time_start)

def oamd_pro_predict(frame,model,count):
    
    tensor = model.predict(frame)
    print('oamd process predict : ',count)
    with lock4:
        process_tensor_dict[count] = tensor

def oamd_paint(model,out):
    num = 0
    while True:
        with lock4:
            if num in process_tensor_dict:
                with lock3:
                    img = process_img_dict[num]
                
                tensor = process_tensor_dict[num]
                img = model.paint(img,tensor,save_image=False)
                out.write(img)
                with lock3:
                    del process_img_dict[num]
              
                del tensor_dict[num]
                print('num :',num)
                num += 1
            elif done_target:
                break
            
# endregion


#one_and_one()
# one_and_multi()       
# one_and_multi_dict()
# two_and_mul()
one_and_mulprocess_dict()
