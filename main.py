from src import model
from src import data
import cv2
import time

if __name__ == '__main__':
    model_id = 1301
    device = 1
    start_time = time.time()
    net=model.load_model(model_id,device)
    #photo
    img0 = cv2.imread('data/img-10002.jpg')
    img1 = data.data_pre_process(img0)
    output = model.predict(net, img1,task=1)
    output = data.data_detect_process(output)
    img2 = data.detect_paint(output,img0)
    cv2.imwrite('data/detect.png', img2)
    # output = data.data_post_process(output)
    # img2=data.data_paint(output,img0,on_original_img=False)
    # cv2.imwrite('data/result.png', img2)
    
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

