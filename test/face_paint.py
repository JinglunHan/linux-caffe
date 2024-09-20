import cv2
import numpy as np
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

image_path = '../data/zhangwei_02.webp'
image_path1 = '../data/zhangwei_03.webp'
img = cv2.imread(image_path)
img3 = cv2.imread(image_path1)
x3_scale = 320/img3.shape[0]
y3_scale = 320/img3.shape[1]
img3 = resize_and_pad(img3, (320,320))
# img3 = cv2.resize(img3,(640,640))

print(img.shape)
x_scale = 320/img.shape[0]
y_scale = 320/img.shape[1]
img1 = cv2.resize(img, (320, 320))
img2 = resize_and_pad(img, (320,320))
box1 = [356.13757 ,92.28067,    223.90219,    290.12073,    416.56406,
  207.26324,    523.7253,     199.36682,    477.99243,    262.10587,
  428.13095,    302.04736,    530.2038,     294.29474]
box2 = [162,  64, 379 ,261]
box3 = [132.69885 ,   85.64597  , 157.56071  , 214.5526,    167.73273,   182.20183,
  244.90858  , 168.25261 ,  214.4661 ,   220.51257,   189.33926,   253.84949,
  256.11172 ,  241.88206  ,   0.9416427]
box1 = [int(x)for x in box1]

box3 = [int(x)for x in box3]
cv2.rectangle(img3, (box1[0],box1[1]), (box1[0]+box1[2], box1[1]+box2[3]), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
for i in range(5):
        # print((int(box1[4+i*2]),int(box1[4+i*2])))
        cv2.circle(img3, (int(box1[4+i*2]*y3_scale),int(box1[5+i*2]*y3_scale)), 5, (0,255,0), -1)
for i in range(5):
        # print((int(box1[4+i*2]),int(box1[4+i*2])))
        cv2.circle(img2, (int(box3[4+i*2]*x_scale+53),int(box3[5+i*2]*x_scale)), 5, (0,255,0), -1)
cv2.rectangle(img2, (int(box3[0]*x_scale+53),int(box3[1]*x_scale)), (int(box3[2]*x_scale+box3[0]*x_scale+53), int(box3[3]*x_scale+box3[1]*x_scale)), (255,255,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.rectangle(img1, (box2[0],box2[1]), (box2[2], box2[3]), (0,255,255), thickness=2, lineType=cv2.LINE_AA)
cv2.namedWindow("SFace Demo", cv2.WINDOW_AUTOSIZE)
cv2.imshow("SFace Demo", img2)
cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()