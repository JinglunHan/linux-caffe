import cv2
import sys
sys.path.append('/home/roota/workstation/onnx2caffe/linux-caffe')
from src import Model
from src import Media
import cv2
import time
import threading
import multiprocessing


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

# region  two_process()

def load_model_and_predict(count_queue,num_queue,process_img_dict,process_tensor_dict):
    print('load_model_and_predict start')
    model = Model(modelid,device)
    while True:
        num = count_queue.get()
        print('load and predict : ',num)
        
        if num == None:
            print('process load and predict done ')
            break
        img = process_img_dict[num]
        tensor = model.predict(img)
        process_tensor_dict[num] = tensor
        num_queue.put(num)
    num_queue.put(None)

def two_process(count_queue,process_img_dict):
    cap = Media(url,1)
    print('two_process start')
    time_start = time.time()
    count = 0
    while cap.media.isOpened():
        ret, frame = cap.media.read()
        print('process add img : ',count)
        if not ret:
            break
        process_img_dict[count] = frame
        count_queue.put(count)
        count += 1
    time.sleep(1)
    count_queue.put(None)

def process_paint(num_queue,process_img_dict,process_tensor_dict):
    global output_path
    print('process_paint start')
    model = Model(modelid,device)
    out = cv2.VideoWriter(output_path+'two_process.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.fps, (cap.width, cap.height))
    num_list = []
    num,count = 0,0
    while True:
        num = num_queue.get()
        print('paint process : ',num,count)
        if num == None and len(num_list)==1:
            print('paint process done ')
            break
        num_list.append(num)
        if count in num_list:
            new_num_list = [num for num in num_list if num != count]
            num_list = new_num_list
            img = process_img_dict[count]
            tensor = process_tensor_dict[count]
            img = model.paint(img,tensor,save_image=False)
            out.write(img)
            del process_img_dict[count]
            del process_tensor_dict[count]
            count += 1





# endregion


if __name__ == '__main__':

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

    process_img_dict = multiprocessing.Manager().dict()
    process_tensor_dict = multiprocessing.Manager().dict()
    count_queue = multiprocessing.Queue()
    num_queue = multiprocessing.Queue()

    # one_and_one()
    # one_and_multi()        
    # one_and_multi_dict()

    pro_img = multiprocessing.Process(target=two_process, args=(count_queue,process_img_dict,))
    # pro_img = threading.Thread(target=two_process, args=(count_queue,cap))
    pro_img.start()

    predict = multiprocessing.Process(target=load_model_and_predict, args=(count_queue,num_queue,process_img_dict,process_tensor_dict,))
    predict.start()
    # # predicts = [multiprocessing.Process(target=load_model_and_predict, args=(count_queue,num_queue)) 
    # #              for _ in range(2)]
    # # for predict in predicts:
    # #     predict.start()

    paint = multiprocessing.Process(target=process_paint, args=(num_queue,process_img_dict,process_tensor_dict,))
    paint.start()

    pro_img.join()
    predict.join()
    paint.join()