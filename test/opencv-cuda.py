import cv2
import time

start = time.time()
# 读取图像
image = cv2.imread('/workdir_test/0_hjl/git/linux-caffe/data/result.jpg')

# 创建 CUDA 图像对象
cuda_image = cv2.cuda_GpuMat()
cuda_image.upload(image)

# 在 GPU 上进行高斯滤波
cuda_blur = cv2.cuda.createGaussianFilter(cuda_image.type(), cuda_image.type(), (0, 0), 2)
cuda_blur.apply(cuda_image, cuda_image)

# 将处理后的图像下载回 CPU
result = cuda_image.download()
cv2.imwrite('/workdir_test/0_hjl/git/linux-caffe/data/fire001_openpose.png', result)

end = time.time()
print('start:',start,'end:',end,'runtime:',end-start)

# import cv2
# import numpy as np

# image_path = './data/fire001_01.png'
# image = cv2.imread(image_path)
# image1 =  cv2.imread(image_path)

# # 定义要画点的坐标
# points = [[260.6734, 273.4473],
#          [263.9691, 268.7800],
#          [255.7253, 268.4341],
#          [  0.0000,   0.0000],
#          [244.7988, 269.9379],
#          [269.1903, 302.4370],
#          [232.5191, 301.8323],
#          [273.2843, 347.3800],
#          [225.0593, 346.0375],
#          [278.0702, 383.8930],
#          [233.2995, 382.7694],
#          [266.4183, 381.1338],
#          [243.8646, 382.6411],
#          [266.7351, 443.7289],
#          [252.2946, 446.5910],
#          [248.5151, 501.7751],
#          [245.9892, 507.3856]]
# points.extend([[(points[6-1][0]+points[7-1][0])/2,(points[6-1][1]+points[7-1][1])/2]])
# points.extend([[(points[12-1][0]+points[13-1][0])/2,(points[12-1][1]+points[13-1][1])/2]])
# print(points)
# for i,p in enumerate(points):
#     p[0] = int(p[0])
#     p[1] = int(p[1])
#     points[i] = tuple(p)
#     print(p)
# print(points)   
# # 在图片上画点
# for point in points:
#     cv2.circle(image1, point, 3, (0, 0, 255), -1)
#     cv2.circle(image, point, 3, (0, 0, 255), -1)  # 5为点的半径，(0, 0, 255)为BGR颜色，-1表示实心圆

# yolov8pose_lines = [(1,2),(1,3),(2,4),(3,5),                      #head
#               (5,7),(4,6),(6,7),(6,12),(7,13),(12,13),      #body
#               (7,9),(9,11),(6,8),(8,10),                    #arm
#               (13,15),(15,17),(12,14),(14,16)]              #leg
# openpose_lines = [(1,2),(1,3),(2,4),(3,5),                      #head
#               (1,18),(6,18),(7,18),(18,19),(12,19),(13,19),      #body
#               (7,9),(9,11),(6,8),(8,10),                    #arm
#               (13,15),(15,17),(12,14),(14,16)]              #leg
# # # 连接指定的点画线
# for i,l in enumerate(yolov8pose_lines):
#     print(l,l[0],points[0])
#     if points[l[0]-1][0] !=0 and points[l[1]-1][0] !=0:
#         cv2.line(image, points[l[0]-1], points[l[1]-1], (255, 0, 0), 1)
#     else:
#         continue  # (255, 0, 0)为BGR颜色，2为线宽
# # 保存新的图片
# cv2.imwrite('./data/fire001_yolopose.png', image)
# #cv2.line(image, points[0], points[8], (255, 0, 0), 1)

# for i,l in enumerate(openpose_lines):
#     print(l,l[0],points[0])
#     if points[l[0]-1][0] !=0 and points[l[1]-1][0] !=0:
#         cv2.line(image1, points[l[0]-1], points[l[1]-1], (255, 0, 0), 1)
#     else:
#         continue
# cv2.imwrite('./data/fire001_openpose.png', image1)