import numpy as np
import os
import onnx
import time
import onnxruntime as ort
import cv2

# model_path = '../models/face_detection_yunet_2023mar.onnx'
model_path = '/home/roota/workstation/onnx2caffe/linux-caffe/models/yolov5n-face.onnx'
#image_path = '/home/roota/workstation/onnx2caffe/linux-caffe/data/fire001_01.png'
# image_path = '../static/face/man_09.jpg'
image_path1 = '../static/face/women_68.jpg'
image_path = '../data/zhangwei_01.webp'
# image_path1 = '../data/zhangwei_05.webp'
print('-----------------___________________--------------------')
print(image_path1,image_path)
def nms_numpy(boxes, scores, iou_threshold):
    # 计算边界框的面积
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # print('scores',scores.shape,scores)
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

def detect_face(image_path,model_path):
    image = cv2.imread(image_path)
    # print('image shape',image.shape)
    x,y = image.shape[1],image.shape[0]
    x_scale = image.shape[1] / 640
    y_scale = image.shape[0] / 640
    img = cv2.resize(image,(640,640))
    image_tensor = np.array([img])
    image_tensor = image_tensor.transpose((0, 3, 1, 2))
    img1 = image_tensor[:,0:1,:,:].copy()
    img2 = image_tensor[:,1:2,:,:].copy()
    img3 = image_tensor[:,2:3,:,:].copy()
    image_tensor = np.concatenate((img3,img2, img1), axis=1)
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.astype(np.float32)
    # print('image_tensor',image_tensor.shape,image_tensor)

    onnx_session = ort.InferenceSession(model_path)
    output = onnx_session.run(None, {"input": image_tensor})
    # print(output[0].shape,output[0])
    # print('cls16',output[1].shape,output[1])
    # print('cls32',output[2].shape,output[2])
    # print('obj8',output[3].shape,output[3])
    # print('obj16',output[4].shape,output[4])
    # print('obj32',output[5].shape,output[5])
    # print('bbox8',output[6].shape,output[6])
    # print('bbox16',output[7].shape,output[7])
    # print('bbox32',output[8].shape,output[8])
    # print('mask8',output[9].shape,output[9])
    # print('mask16',output[10].shape,output[10])
    # print('mask32,',output[11].shape,output[11])

    result = output[0][0]
    # print('result',result.shape)
    xconf = result[:,4]
    xconf = xconf.reshape(25200,1)
    # print('xconf',xconf.shape,xconf)
    conf = result[:,4] > 0.5
    xconf = xconf[conf,:]
    # print('conf',conf.shape,conf)
    xc = result[conf,15]
    # print('xc',xc.shape,xc)
    x = result[:,0]
    y = result[:,1]
    w = result[:,2]
    h = result[:,3]
    result[:,0] = x-(w/2)
    result[:,1] = y-(h/2)
    result[:,2] = x+(w)
    result[:,3] = y+(h)
    result = result[conf,:]
    boxes = result[:,:4]
    boxes = boxes.astype(int)
    # print('boxes',boxes.shape,boxes)
    l = nms_numpy(boxes,xconf,0.2)
    # print('l',l.shape,l,l[0])
    x1 = int(boxes[l[0],0] * x_scale)
    # print(x1)
    y1 = int(boxes[l[0],1] * y_scale)
    x2 = int(boxes[l[0],2] * x_scale)
    y2 = int(boxes[l[0],3] * y_scale)
    face_image = image[y1:y2,x1:x2]
    # print(boxes[l[0],:])
    result = result[l[0],:]
    resultp1 = result[0:4]
    resultp2 = result[5:15]
    resultp = np.concatenate((resultp1,resultp2),axis=0)
    for i in range(7):
        resultp[i*2+1] = resultp[i*2+1] * y_scale
        resultp[i*2] = resultp[i*2] * x_scale
    # print('result',result.shape,result)
    # print('resultp',resultp.shape,resultp)
    # cv2.rectangle(image, (x1,y1), (x2, y2), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
    # for i in range(5):
        # print((int(result[i*2]* x_scale),int(result[i*2]* y_scale)))
        # cv.circle(image, (int(result[5+i*2]* x_scale),int(result[6+i*2]* y_scale)), 5, (0,255,i*50), -1)
    # cv2.imwrite('/home/roota/workstation/onnx2caffe/linux-caffe/test/test_data/face'+str(time.time())+'.jpg', face_image)
    return face_image,resultp2 

def resize_and_pad(img, size, pad_color=0):
    """
    Resize and pad an image to a target size while maintaining the aspect ratio.
    
    Parameters:
    img_path (str): Path to the image file.
    size (tuple): Target size (width, height) in pixels.
    pad_color (int, tuple): Color for padding areas (default is black).
    
    Returns:
    numpy.ndarray: Resized and padded image.
    """

    # Calculate aspect ratios
    h, w = img.shape[:2]
    target_ratio = size[0] / size[1]
    actual_ratio = w / h
    
    # Determine scaling and new dimensions
    if actual_ratio > target_ratio:  # Width is larger, scale to width
        new_w = size[0]
        new_h = int(new_w / actual_ratio)
    else:  # Height is larger, scale to height
        new_h = size[1]
        new_w = int(new_h * actual_ratio)
    
    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a blank image of the target size
    result_img = np.full((size[1], size[0], 3), pad_color, dtype=np.uint8)
    
    # Calculate the position to place the resized image
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2
    
    # Place the resized image on the blank image
    result_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return result_img

def get_similarity_transform_matrix(src):
    # Define the destination landmarks (mean face landmarks)
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # Compute mean of src and dst.
    src_mean = np.mean(src, axis=0)
    dst_mean = np.array([56.0262, 71.9008], dtype=np.float32)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Compute the transformation matrix using SVD
    A = np.dot(dst_demean.T, src_demean) / 5
    U, S, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, 1] *= -1
        R = np.dot(U, Vt)

    # Compute scaling factor and translation
    var1 = np.var(src_demean[:, 0])
    var2 = np.var(src_demean[:, 1])
    scale = 1.0 / (var1 + var2) * (S[0] + S[1])
    t = dst_mean - scale * np.dot(R, src_mean)

    # Compose the transformation matrix
    transform_mat = np.eye(3)
    transform_mat[:2, :2] = scale * R
    transform_mat[:2, 2] = t
    transform_mat = transform_mat[:2,:]
    print('transform_mat',transform_mat)
    return transform_mat
# face_image = image[y1:y2,x1:x2]
# cv2.imwrite('/home/roota/workstation/onnx2caffe/linux-caffe/test/test_data/face_cut.jpg',face_image)

model_path1 = '/home/roota/workstation/onnx2caffe/linux-caffe/models/face_recognition_sface_2021dec_int8.onnx'
model_path1 = '/home/roota/workstation/onnx2caffe/linux-caffe/models/face_recognition_sface_2021dec.onnx'
model_path2 = '/home/roota/workstation/onnx2caffe/linux-caffe/models/mobilefacenet.onnx'
model_path3 = '/home/roota/workstation/onnx2caffe/linux-caffe/models/model_r18.onnx'
def recognize_face(model_path1,face_image):
    face_image = cv2.resize(face_image,(112,112))
    face_tensor = np.array([face_image])

    onnx_session2 = ort.InferenceSession(model_path1)

    face_tensor = face_tensor.transpose((0, 3, 1, 2))
    # print('face tensor',face_tensor.shape)
    face_img1 = face_tensor[:,0:1,:,:].copy()
    face_img2 = face_tensor[:,1:2,:,:].copy()
    face_img3 = face_tensor[:,2:3,:,:].copy()
    face_tensor = np.concatenate((face_img3,face_img2, face_img1), axis=1)
    face_tensor = face_tensor / 255.0
    face_tensor = face_tensor.astype(np.float32)
    input_name = onnx_session2.get_inputs()[0].name
    # print('face tensor',face_tensor.shape)
    face_recognanize = onnx_session2.run(None, {input_name : face_tensor})
    # face_recognanize = np.array(face_recognanize)
    # print(face_recognanize.shape,face_recognanize)
    return face_recognanize[0]

def align_face(img,landmarks):
    # Define the destination landmarks (mean face landmarks)
    dx = landmarks[0] - landmarks[2]
    dy = landmarks[1] - landmarks[3]
    
    eye_center = ((landmarks[0]+landmarks[2]) // 2,(landmarks[1]+landmarks[3]) // 2)
    angle = np.degrees(np.arctan2(dy,dx))
    if angle > 90:
        angle =angle - 180
    elif angle < -90:
        angle = angle + 180
    print('angle',angle)
    rotation_matrix = cv2.getRotationMatrix2D(center=(img.shape[1]//2, img.shape[0]//2), angle=angle, scale=1)
    rotated_img = cv2.warpAffine(img, M=rotation_matrix, dsize=(img.shape[1], img.shape[0]))#, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img
 
def match(face_feature1, face_feature2, dis_type):
    # 确保输入是numpy数组
    face_feature1 = np.array(face_feature1)
    face_feature2 = np.array(face_feature2)
    
    # 归一化特征向量
    face_feature1 = face_feature1 / np.linalg.norm(face_feature1)
    face_feature2 = face_feature2 / np.linalg.norm(face_feature2)

    if dis_type == 'cosine'or dis_type == 0:
        # 计算余弦相似度
        similarity = np.sum(face_feature1 * face_feature2)
        
    elif dis_type == 'euclidean' or dis_type == 1:  # L2范数距离等同于欧氏距离
        # 计算L2范数距离
        distance = np.linalg.norm(face_feature1 - face_feature2)
        similarity = -distance  # 为了与相似度概念一致，取负数
        
    else:
        raise ValueError(f"Invalid distance type: {dis_type}")

    return similarity

import cv2 as cv
face1,output1 = detect_face(image_path,model_path)
face2,output2 = detect_face(image_path1,model_path)
# image1 = cv2.imread(image_path)
# image2 = cv2.imread(image_path1)
# img2 = resize_and_pad(image1, (320,320))
# print(output1.shape,output1)
# output1 = np.resize(output1,(5,2))
# print(output1.shape,output1)
# trans_output1 = get_similarity_transform_matrix(output1)
# print(trans_output1.shape)
face1 = align_face(face1,output1)
face2 = align_face(face2,output2)

# cv2.imshow('Aligned Image', wrap_output1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
cv2.imwrite('/home/roota/workstation/onnx2caffe/linux-caffe/test/test_data/face1'+str(time.time())+'.jpg', face1)
cv2.imwrite('/home/roota/workstation/onnx2caffe/linux-caffe/test/test_data/face2'+str(time.time())+'.jpg', face2)
encode1 = recognize_face(model_path3,face1)
encode2 = recognize_face(model_path3,face2)
print('model',os.path.basename(model_path3))
# print('encode1',encode1.shape,encode1,'\n','encode2',encode2.shape,encode2)
result1 = np.sum(encode1[0] * encode2[0])
result2 = np.linalg.norm(encode1[0]) - np.linalg.norm(encode2[0])
# result1 = cv.FaceRecognizerSF.match(face_feature1=encode1,face_feature2=encode2,dis_type='FR_COSINE')
# result2 = cv.FaceRecognizerSF.match(face_feature1=encode1,face_feature2=encode2,dis_type='FR_NORM_L2')
print('result1:',result1,'result2:',result2)
dot = np.dot(encode1[0],encode2[0])
nrom = np.linalg.norm(encode1[0]-encode2[0])
print('dot:',dot,'nrom:',nrom)
match1 = match(encode1,encode2,0)
match2 = match(encode1,encode2,1)
print('match1:',match1,'match2:',match2)

import json
json_data = json.dumps(encode1[0].tolist())
print(json_data)
data = json.loads(json_data)
print(data)
print(type(data))