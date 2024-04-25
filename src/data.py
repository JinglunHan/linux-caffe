import numpy as np
import cv2

conf_thres = 0.5
iou_thres = 0.25

def data_pre_process(img):
    img = cv2.resize(img, (640, 640))
    img_tensor = np.array([img])
    # print(img_tensor.shape)
    img_tensor = img_tensor.transpose((0, 3, 1, 2))
    img1 = img_tensor[:,0:1,:,:].copy()
    img2 = img_tensor[:,1:2,:,:].copy()
    img3 = img_tensor[:,2:3,:,:].copy()
    img_tensor = np.concatenate((img3, img2, img1), axis=1)
    img_tensor = img_tensor.astype('float32') / 255.0
    return img_tensor

def data_post_process(tensor):
    tensor = tensor.reshape((1,57,8400))
    tensor = tensor.transpose((0,2,1))
    tensor = xywh2xyxy(tensor)
    conf = tensor[:,:,2:3] > conf_thres*100
    conf = conf.reshape((1,8400))
    output = [np.zeros((57))]
    # print(conf.shape,conf)
    for i , j in enumerate(tensor):
        # print('i: j:',i,j.shape)
        j = j[conf[i]]
        # print('i: j:',i,j.shape)
        scores = j[:,2:3].astype(int)
        boxes = np.concatenate((j[:,0:2],j[:,3:5]),axis = 1)
        l = nms_numpy(boxes,scores,iou_thres)
        # print(l.shape,l)
        # print('j[l]:',j[l].shape,j[l])
        output[i] = j[l]
        # print(output)
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

#draw box and line 
def data_paint(tensor,img,
               paint_box=True,
               box_color=(255,0,0),
               box_weight=2,
               pose_kind=0,
               line_color=(0,0,255),
               line_weight=5):
    
    tensor = np.array(tensor)
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
        points[i][0:51] = j[6:57]
        # add points 18,19
        points[i][51] = (j[21] + j[24])/2
        points[i][52] = (j[22] + j[25])/2
        points[i][53] = (j[23] + j[26])/2
        points[i][54] = (j[39] + j[42])/2
        points[i][55] = (j[40] + j[43])/2
        points[i][56] = (j[41] + j[44])/2
        
    
    weight = img.shape[1]
    height = img.shape[0]
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

    for i,j in enumerate(points):
        if pose_kind == 0:
            for x in yolov8pose_lines:
                if j[x[0]*3-1] >conf_thres*100 and j[x[1]*3-1] >conf_thres*100:
                    cv2.line(img,[int(j[x[0]*3-3]*weight/640),int(j[x[0]*3-2]*height/640)],
                             [int(j[x[1]*3-3]*weight/640),int(j[x[1]*3-2]*height/640)],line_color,line_weight)
        if pose_kind == 1:
            for x in openpose_lines:
                if j[x[0]*3-1] >conf_thres*100 and j[x[1]*3-1] >conf_thres*100:
                    cv2.line(img,[int(j[x[0]*3-3]*weight/640),int(j[x[0]*3-2]*height/640)],
                             [int(j[x[1]*3-3]*weight/640),int(j[x[1]*3-2]*height/640)],line_color,line_weight)
    return img


            

