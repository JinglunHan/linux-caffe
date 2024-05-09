import numpy as np
import cv2

# 创建一个带有透明背景的空白图像
height, width = 300, 400
blank_image = np.zeros((height, width, 4), dtype=np.uint8)

# 设置线条颜色和线条宽度
color = (0, 255, 0,255)  # 线条颜色为绿色
thickness = 2  # 线条宽度为2个像素

# 在图像上绘制不透明的线条
start_point = (50, 50)
end_point = (350, 250)
cv2.line(blank_image, start_point, end_point, color, thickness)

# 保存绘制好的图像
cv2.imwrite("opaque_line.png", blank_image)