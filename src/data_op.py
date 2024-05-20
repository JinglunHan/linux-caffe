import numpy as np
import cv2
# import cupy as cp
import time

conf_thres = 0.3
iou_thres = 0.25
device = 0      # 0:cpu 1:gpu

num = 0

def data_pre_process(img,task):
    start_time = time.time()
    if len(img)==0:
        return 0
    if task == 0:
        img = cv2.resize(img, (640, 352))
    elif task == 1:
        img = cv2.resize(img, (640, 640))
    if device == 0:
        img_tensor = np.array([img])
    else:
        img_tensor = cp.array([img])
    # print(img_tensor.shape)
    img_tensor = img_tensor.transpose((0, 3, 1, 2))
    img1 = img_tensor[:,0:1,:,:].copy()
    img2 = img_tensor[:,1:2,:,:].copy()
    img3 = img_tensor[:,2:3,:,:].copy()
    if device == 0:
        img_tensor = np.concatenate((img3, img2, img1), axis=1)
    else:
        img_tensor = cp.concatenate((img3, img2, img1), axis=1)
    img_tensor = img_tensor.astype('float32') / 255.0
    end_time = time.time()
    # global num 
    # if num < 3:
    #     print('data pre process : ',end_time-start_time)
    return img_tensor

### pose model result
def data_post_process(tensor):
    start_time = time.time()
    if device == 1:
        tensor = cp.array(tensor)
        output = [cp.zeros((57))]
    else:
        output = [np.zeros((57))]
    tensor = tensor.reshape((1,57,8400))
    tensor = tensor.transpose((0,2,1))
    tensor = xywh2xyxy(tensor)
    conf = tensor[:,:,2:3] > conf_thres*100
    conf = conf.reshape((1,8400))
    # print(conf.shape,conf)
    for i , j in enumerate(tensor):
        # print('i: j:',i,j.shape)
        j = j[conf[i]]
        # print('i: j:',i,j.shape)
        scores = j[:,2:3].astype(int)
        if device == 0:
            boxes = np.concatenate((j[:,0:2],j[:,3:5]),axis = 1)
            l = nms_numpy(boxes,scores,iou_thres)
        else:
            boxes = cp.concatenate((j[:,0:2],j[:,3:5]),axis = 1)
            l = nms_cupy(boxes,scores,iou_thres)
        if len(l) ==0:        
            # print(l.shape,l)
            return [[[-1]]]
        # print('j[l]:',j[l].shape,j[l])
        else:
            output[i] = j[l]
        print(output[0].shape)
    end_time = time.time()
    # global num 
    # if num < 3:
    #     print('data post process : ',end_time-start_time)
    return output


def xywh2xyxy(x):
    dw = x[..., 3] / 2  # half-width
    dh = x[..., 4] / 2  # half-height
    x[...,3] = x[...,0] + dw
    x[...,4] = x[...,1] + dh
    x[...,0] = x[...,0] - dw
    x[...,1] = x[...,1] - dh
    return x

def nms_numpy(boxes, scores, iou_threshold):
    # 计算边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # print(scores.shape,scores)
    # print(scores.argsort)
    # 按照分数降序排列
    order = scores.argsort(axis=0)[::-1]
    # print(order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i[0])

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # 计算交集面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # 计算IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 找到IoU小于阈值的边界框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    # print('keep:',keep)
    return np.array(keep)

def nms_cupy(boxes, scores, iou_threshold):
    # 计算边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # print(scores.shape,scores)
    # print(scores.argsort)
    # 按照分数降序排列
    order = scores.argsort(axis=0)[::-1]
    # print(order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i[0])

        xx1 = cp.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = cp.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = cp.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = cp.minimum(boxes[i, 3], boxes[order[1:], 3])

        # 计算交集面积
        w = cp.maximum(0.0, xx2 - xx1 + 1)
        h = cp.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h

        # 计算IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 找到IoU小于阈值的边界框
        inds = cp.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    # print('keep:',keep)
    return cp.array(keep)

#draw box and line 
def data_paint(tensor,img,
               paint_box=True,
               box_color=(255,0,0),
               box_weight=2,
               pose_kind=0,
               line_color=(0,0,255),
               line_weight=3,
               on_original_img=True):
    start_time = time.time()
    if device == 0:
        tensor = np.array(tensor)
    else:
        cp_tensor = cp.array(tensor)
        tensor = cp_tensor.get()
    bc = tensor.shape[1]
    # print('tensor',tensor)
    boxes = [np.zeros((6))]*bc
    points = [np.zeros((57))]*bc

    
    ### pose kind = 0
    yolov8pose_lines = [(1,2),(1,3),(2,4),(3,5),                      #head
              (5,7),(4,6),(6,7),(6,12),(7,13),(12,13),      #body
              (7,9),(9,11),(6,8),(8,10),                    #arm
              (13,15),(15,17),(12,14),(14,16)]              #leg
    ### pose kind =1
    openpose_lines = [(1,2),(1,3),(2,4),(3,5),                      #head
                (1,18),(6,18),(7,18),(18,19),(12,19),(13,19),      #body
                (7,9),(9,11),(6,8),(8,10),                    #arm
                (13,15),(15,17),(12,14),(14,16)]              #leg
    
    for i , j in enumerate(tensor[0]):
        # print('i:',i,'\n','j:',j)
        boxes[i] = j[0:6]
        # print('boxes:',boxes)
      
        points[i] = j[6:57]
        # print('points[i]:',i,points[i])

    
    weight = img.shape[1]
    height = img.shape[0]
    if on_original_img == False:
        img = np.zeros((height, weight, 4), dtype=np.uint8)
        box_color = (*box_color,255)
        line_color = (*line_color,255)
    # print('weight height',weight,height,img.shape)
    if paint_box:
        for i ,j in enumerate(boxes):
            # print(i,j.shape)
            j[0] = int(j[0]*weight/640)
            j[1] = int(j[1]*height/640)
            j[3] = int(j[3]*weight/640)
            j[4] = int(j[4]*height/640)
            j=j.astype(int)
            # print(j,j[0])
            cv2.line(img,[j[0],j[1]],[j[0],j[4]],box_color,box_weight)
            cv2.line(img,[j[0],j[1]],[j[3],j[1]],box_color,box_weight)
            cv2.line(img,[j[3],j[4]],[j[3],j[1]],box_color,box_weight)
            cv2.line(img,[j[0],j[4]],[j[3],j[4]],box_color,box_weight)
    # print(points)
    if pose_kind == 0:
        for i,j in enumerate(points):
            # print("i:",i,'j:',j)
            for x in yolov8pose_lines:
                if j[x[0]*3-1] >conf_thres*100 and j[x[1]*3-1] >conf_thres*100:
                    cv2.line(img,[int(j[x[0]*3-3]*weight/640),int(j[x[0]*3-2]*height/640)],
                             [int(j[x[1]*3-3]*weight/640),int(j[x[1]*3-2]*height/640)],line_color,line_weight)
    if pose_kind == 1:
        for i ,j in enumerate(points):
            j = np.append(j,(j[15] + j[18])/2)
            j = np.append(j,(j[16] + j[19])/2)
            j = np.append(j,(j[17] + j[20])/2)
            j = np.append(j,(j[33] + j[36])/2)
            j = np.append(j,(j[34] + j[37])/2)
            j = np.append(j,(j[35] + j[38])/2)
            points[i] = j
        # print(points[0].shape,points)
        
        for i,j in enumerate(points):
            for x in openpose_lines:
                if j[x[0]*3-1] >conf_thres*100 and j[x[1]*3-1] >conf_thres*100:
                    cv2.line(img,[int(j[x[0]*3-3]*weight/640),int(j[x[0]*3-2]*height/640)],
                             [int(j[x[1]*3-3]*weight/640),int(j[x[1]*3-2]*height/640)],line_color,line_weight)
    
    end_time = time.time()
    global num 
    if num < 3:
        print('paint : ',end_time-start_time)
        num = num + 1
    # print('paint : ',end_time-start_time)
    return img


##### detect model result
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
def data_detect_process(tensor):
    nc = int(tensor[0].shape[1]/3-5)
    nw = nc+5
    img_width,img_height = 640,352
    boxes = np.zeros((1,4))
    scores = np.zeros((1,1))
    output = np.zeros((1,nw))
    for i , x in enumerate(tensor):
        x_width = x.shape[3]
        x_height = x.shape[2]

        x = x.reshape((1,(nc+5)*3,-1))
        x = x.transpose((0,2,1))

        for j in range(3):
            y = x[:,:,nw*j:nw*(j+1)]

            #xywh 
            x_dst = 2*sigmoid(y[:,:,0])-0.5
            y_dst = 2*sigmoid(y[:,:,1])-0.5

            for xh in range(x_height):
                for xw in range(x_width):
                    x_dst[0][xh*x_width+xw] += xw
                    y_dst[0][xh*x_width+xw] += xh

            x_dst = x_dst / x_width
            y_dst = y_dst / x_height
            w_dst = sigmoid(y[:,:,2])
            h_dst = sigmoid(y[:,:,3])
            w_dst = (2*w_dst)*(2*w_dst)*(anchors[6*i+2*j])/img_width
            h_dst = (2*h_dst)*(2*h_dst)*(anchors[6*i+2*j+1])/img_height
            #xywh to xyxy
            y[:,:,0] = (x_dst - w_dst/2)*img_width
            y[:,:,1] = (y_dst - h_dst/2)*img_height
            y[:,:,2] = (x_dst + w_dst/2)*img_width
            y[:,:,3] = (y_dst + h_dst/2)*img_height
            y[:,:,0][y[:,:,0]<0] = 0
            y[:,:,1][y[:,:,1]<0] = 0
            y[:,:,2][y[:,:,2]<0] = 0
            y[:,:,3][y[:,:,3]<0] = 0
            y[:,:,0][y[:,:,0]>img_width] = img_width
            y[:,:,1][y[:,:,1]>img_height] = img_height
            y[:,:,2][y[:,:,2]>img_width] = img_width
            y[:,:,3][y[:,:,3]>img_height] = img_height
                       
            conf = sigmoid(y[:,:,4]) > conf_thres
            y[:,:,4:] = sigmoid(y[:,:,4:])

            y = y[0][conf[0]]
            output = np.concatenate((output,y),axis=0)
            boxes = np.concatenate((boxes,y[:,:4]),axis=0)
            scores = np.concatenate((scores,y[:,4:5]))

    output = output[1:,:]
    boxes = boxes[1:,:]
    scores = scores[1:,:]
    l = nms_numpy(boxes,scores,iou_thres)
    # print(l.shape,len(l),l)
    if len(l)==0:
        return [[-1]]
    output = output[l]
    # print(output.shape)
    
    return output

def detect_paint(tensor,img,box_line=3):
    
#     colors_rgb = {
#     'red': (255, 0, 0),
#     'green': (0, 255, 0),
#     'blue': (0, 0, 255),
#     'yellow': (255, 255, 0),
#     'orange': (255, 165, 0),
#     'purple': (128, 0, 128),
#     'cyan': (0, 255, 255),
#     'magenta': (255, 0, 255),
#     'lime': (0, 255, 0),
#     'pink': (255, 192, 203),
#     'brown': (165, 42, 42),
#     'teal': (0, 128, 128),
#     'navy': (0, 0, 128),
#     'maroon': (128, 0, 0),
#     'olive': (128, 128, 0),
#     'gray': (128, 128, 128),
#     'black': (0, 0, 0),
#     'white': (255, 255, 255),
#     'gold': (255, 215, 0),
#     'silver': (192, 192, 192)
# }
    colors_space = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 165, 0),(128, 0, 128),(0, 255, 255),(255, 0, 255),
                    (0, 255, 0),(255, 192, 203),(165, 42, 42),(0, 128, 128),(0, 0, 128),(128, 0, 0),(128, 128, 0),(128, 128, 128),
                    (0, 0, 0),(255, 255, 255),(255, 215, 0),(192, 192, 192)]
    
    nc = tensor.shape[1]-5
    weight = img.shape[1]
    height = img.shape[0]
    img_width,img_height = 640,352
    class_id,class_conf,max_class_conf = 0,0,0
    for i,x in enumerate(tensor):
        x1 = int(x[0]/img_width*weight)
        y1 = int(x[1]/img_height*height)
        x2 = int(x[2]/img_width*weight)
        y2 = int(x[3]/img_height*height)
        for n in range(nc):
            class_conf = x[5+n]
            if max_class_conf < class_conf:
                max_class_conf = class_conf
                class_id = n
        cv2.line(img,[x1,y1],[x1,y2],colors_space[class_id],box_line)
        cv2.line(img,[x1,y2],[x2,y2],colors_space[class_id],box_line)
        cv2.line(img,[x2,y2],[x2,y1],colors_space[class_id],box_line)
        cv2.line(img,[x2,y1],[x1,y1],colors_space[class_id],box_line)
        max_class_conf,class_id = 0,0
    
    return img
            

