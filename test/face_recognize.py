import face_recognition
import cv2
import numpy as np

path = '/home/roota/workstation/onnx2caffe/linux-caffe/static/face/man_01.jpg'
# image = face_recognition.load_image_file(path)
image = cv2.imread(path)
print(image)
image1 = image[:, :, ::-1]
print(image1)
location = face_recognition.face_locations(image1)
encodeing = face_recognition.face_encodings(image1,location)
print(encodeing)
