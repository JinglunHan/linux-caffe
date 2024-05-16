# from src import model
# from src import data
from src import Model
from src import Media
import cv2
import time

if __name__ == '__main__':
    model_id = 1001
    device = 0
    start_time = time.time()
    url = 'rtsp://admin:admin12345@192.168.6.64:554/Streaming/Channels/103?transportmode=unicast&profile=Profile_3'
    url1 = 'rtsp://192.168.2.198:8554/micagent1'
    # net=model.load_model(model_id,device)
    #photo
    img0 = cv2.imread('data/fire001_01.png')
    cap=Media(url1,1)
    cap.write_video()
    # img1 = data.data_pre_process(img0)
    # output = model.predict(net, img1,task=1)
    # output = data.data_detect_process(output)
    # img2 = data.detect_paint(output,img0)
    # cv2.imwrite('data/detect.png', img2)

    # output = data.data_post_process(output)
    # img2=data.data_paint(output,img0,on_original_img=False)
    # cv2.imwrite('data/result.png', img2)
    
    # caffe_model = Model(model_id, device)
    # output = caffe_model.predict(img0)
    # caffe_model.paint(img0,output)

    # #video
    # out_path = 'data/x_move.mp4'
    # cap = cv2.VideoCapture('data/move.mp4')
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # # Create a VideoWriter object to write the output video
    # out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    # while cap.isOpened():
    #     ret,frame = cap.read()
    #     if ret:
    #         # print('frame :',ret)
    #         img = data.data_pre_process(frame)
    #         output = model.predict(net, img)
    #         output = data.data_post_process(output)
    #         img = data.data_paint(output, frame)
    #         out.write(img)
    #     else:
    #         break
    
    # #stream
    # rtsp_url = 'rtsp://192.168.2.198:8554/micagent1'
    # cap = cv2.VideoCapture(rtsp_url)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         img = data.data_pre_process(frame)
    #         output = model.predict(net, img)
    #         output = data.data_post_process(output)
    #         img = data.data_paint(output, frame)
    #         cv2.imshow('frame', img)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # cap.release()
    # cv2.destroyAllWindows()

    end_time = time.time()
    print('runtime:',end_time-start_time)

