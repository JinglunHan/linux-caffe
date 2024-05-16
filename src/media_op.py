import cv2
import numpy as np
import os
from datetime import datetime

current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_path)
abs_path = os.path.dirname(parent_directory)

class Media():
    def __init__(self,form,type):
        if type == 'image' or type ==0:
            self.media = self.read_image(form)
            self.width = self.media.shape[1]
            self.height = self.media.shape[0]
        elif type == 'video' or type ==1:
            self.media = self.read_video(form)
            self.url = form
            self.width = int(self.media.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.media.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.media.get(cv2.CAP_PROP_FPS))
        

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        return image

    def read_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        return video

    def write_video(self, video_path=abs_path+'/record/',ip_address='localhost'):
        time = datetime.now()
        target_dir = video_path+time.strftime("%Y%m%d")+'/'
        os.makedirs(target_dir, exist_ok=True)
        video_path = target_dir+time.strftime("%H%M%S")+'temp.mp4'
        result_path = target_dir+'IP_'+ip_address+'_'+time.strftime("%H%M%S")+'.mp4'
        print(result_path)
        # codec = cv2.VideoWriter_fourcc(*'mp4v')
 
        # out = cv2.VideoWriter(video_path, codec, self.fps, (self.width, self.height),True)
        # os.system('ffmpeg -i {} -c:v libx264  -c:a aac -strict -2 {}'.format(self.url,result_path))
        os.system('/home/roota/workstation/opencv/ffmpeg/bin/ffmpeg -i {} -c:v libx264 -preset medium -crf 23 -c:a aac -strict -2 -y {}'.format(self.url,result_path))
            # -c:v libx264: 指定视频编码器为libx264。
            # -preset medium: 设置视频编码的预设速度/质量平衡为medium，你可以根据需要选择不同的预设值，例如ultrafast、fast、slow等。
            # -crf 23: 指定视频的质量，23表示中等质量，数值越小质量越高。
            # -c:a aac: 指定音频编码器为AAC。
            # -strict -2: 设置音频编码器的兼容性级别。

        # while(True):
        #     ret, frame = self.media.read()
        #     if ret:
        #         out.write(frame)
        #     else:
        #         break