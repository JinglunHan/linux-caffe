import cv2
import numpy as np
import onnxruntime
import src.data_op as dop

detect_model_path = './models/yolov5n-face.onnx'
recognize_model_path = './models/model_r18.onnx'
detect_threshold = 0.5
recognize_threshold = 0.4
iou_thres = 0.25

class FaceRecognizer:

    def __init__(self,task=1,face_owner_name=None):
        '''
        initialize the face recognizer;
        task('0'input face data,'1'detect face image),default is '1';
        face_owner_name: the name of the owner of the face image(if task is '0'),default is None;
        '''
        self.x_scale ,self.y_scale = 1.0,1.0
        self.task = task
        self.name = face_owner_name
        self.detect_threshold = detect_threshold
        self.recognize_threshold = recognize_threshold
        self.detect_session = onnxruntime.InferenceSession(detect_model_path)
        self.recognize_session = onnxruntime.InferenceSession(recognize_model_path)

    def detect_prepare(self, image):
        '''
        data prepare for detect model
        input: image, output: image_tensor
        '''
        
        print('---------------------image shape:-------------',image.shape)
        self.x_scale = image.shape[1] / 640.0
        self.y_scale = image.shape[0] / 640.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image_tensor = np.array([image])
        image_tensor = image_tensor.transpose((0, 3, 1, 2))
        img1 = image_tensor[:,0:1,:,:].copy()
        img2 = image_tensor[:,1:2,:,:].copy()
        img3 = image_tensor[:,2:3,:,:].copy()
        image_tensor = np.concatenate((img3,img2, img1), axis=1)
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.astype(np.float32)
        return image_tensor
    
    def recognize_prepare(self, image):
        '''
        data prepare for recognize model
        input: image, output: face_tensor
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(image,(112,112))
        face_tensor = np.array([face_image])
        face_tensor = face_tensor.transpose((0, 3, 1, 2))
        face_img1 = face_tensor[:,0:1,:,:].copy()
        face_img2 = face_tensor[:,1:2,:,:].copy()
        face_img3 = face_tensor[:,2:3,:,:].copy()
        face_tensor = np.concatenate((face_img3,face_img2, face_img1), axis=1)
        face_tensor = face_tensor / 255.0
        face_tensor = face_tensor.astype(np.float32)
        return face_tensor
    
    def detect_process(self, image):
        '''
        detect face in image
        input: image
        output: face_image_list,land_mark_list
        '''
        face_image_list = []
        land_mask_list = []
        image_tensor = self.detect_prepare(image)
        #use onnxruntime to run model
        input_name = self.detect_session.get_inputs()[0].name
        output = self.detect_session.run(None, {input_name: image_tensor})

        result = output[0][0]
        xconf = result[:,4]
        xconf = xconf.reshape(25200,1)
        conf = result[:,4] > self.detect_threshold
        xconf = xconf[conf,:]
        xc = result[conf,15]
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
        l = dop.nms_numpy(boxes,xconf,iou_thres)

        if len(l) == 0:
            return [[-1]],[[-1]]
        else:
            for n,i in enumerate(l):
                x1 = int(boxes[i,0]*self.x_scale)
                y1 = int(boxes[i,1]*self.y_scale)
                x2 = int(boxes[i,2]*self.x_scale)
                y2 = int(boxes[i,3]*self.y_scale)
                face_image = image[y1:y2,x1:x2]
                face_image_list.append(face_image)
                for j in range(5):
                    result[i,5+j*2]*self.x_scale
                    result[i,5+j*2+1]*self.y_scale
                land_mark = result[i,5:15]
                land_mask_list.append(land_mark)
                if self.task == 0:
                    cv2.imwrite('./static/face/'+str(self.name)+'.jpg',face_image)
                    # if want to save face image according to groups ,make new dir for each group,and add group in class
                    # os.makedirs('./static/face_data/'+str(self.group))
                    # cv2.imwrite('./static/face_data/'+str(self.group)+'/'+str(self.name)+'.jpg',face_image)
                elif self.task == 1:
                    cv2.imwrite('./static/temp/'+str(n)+'.jpg',face_image)
            # print('-------- face_image_list:',len(face_image_list),face_image_list[0].shape,len(face_image_list[0]),'---------')
            return face_image_list,land_mask_list


    def recognize_process(self, image_list,land_mask_list):
        '''
        recognize face in image and output face encode
        input: image_list,land_mark_list
        output: face_encode_list
        '''
        face_encode_list = []
        for n,i in enumerate(image_list):
            face = align_face(i,land_mask_list[n])
            face_tensor = self.recognize_prepare(face)
            #use onnxruntime to run model
            input_name = self.recognize_session.get_inputs()[0].name    
            output = self.recognize_session.run(None, {input_name: face_tensor})
            face_encode = output[0]
            face_encode_list.append(face_encode)
        return face_encode_list
    
    def enter_data(self,image,face_owner_name):
        '''
        enter face data and face owner name,output face encode which should be stored to db
        '''
        self.task = 0
        self.name = face_owner_name
        face_image_list,land_mask_list = self.detect_process(image)
        if len(face_image_list[0]) == 1:
            return [[-1]]
        face_encode_list = self.recognize_process(face_image_list,land_mask_list)
        return face_encode_list
    

    def compare_face_data(self,image,encode_list):
        '''
        input detect image and encode_list from face db
        output: list_index(face detected in encode list),face_list(face detect in image)
        '''
        list_index = []
        self.task = 1
        face_image_list,land_mask_list = self.detect_process(image)
        face_encode_list = self.recognize_process(face_image_list,land_mask_list)
        for m,i in enumerate(face_encode_list):  
            for n,j in enumerate(encode_list):
                similarity = match(j,i)
                print(m,n,'similarity:',similarity)
                if similarity > self.recognize_threshold:
                    list_index.append(n)
                    break
            if len(list_index)< m + 1:
                list_index.append(-1) 

        
        return list_index,face_image_list


def match(face_feature1, face_feature2, dis_type=0):
    '''
    -- input:face encode1 and face encode2, dis_type,-- output:similarity ; ||
    compare face encode,according to dis_type choose the way of campare ; ||
    dis_type default value = 0, similarity threshold > 0.4 ; 
    dis_type = 1, similarity threshold < 1.1 (this way is not perfect)
    '''
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
        similarity = distance  # 为了与相似度概念一致，取负数  
    else:
        raise ValueError(f"Invalid distance type: {dis_type}")

    return similarity

def align_face(img,landmarks):
    '''
    rotate face to align face
    input: image,landmarks
    output: rotated image
    '''
    dx = landmarks[0] - landmarks[2]
    dy = landmarks[1] - landmarks[3]
    
    eye_center = ((landmarks[0]+landmarks[2]) // 2,(landmarks[1]+landmarks[3]) // 2)
    angle = np.degrees(np.arctan2(dy,dx))
    if angle > 90:
        angle =angle - 180
    elif angle < -90:
        angle = angle + 180
    # print('angle',angle)
    rotation_matrix = cv2.getRotationMatrix2D(center=(img.shape[1]//2, img.shape[0]//2), angle=angle, scale=1)
    rotated_img = cv2.warpAffine(img, M=rotation_matrix, dsize=(img.shape[1], img.shape[0]))#, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img