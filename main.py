from src import model
from src import data
import cv2
import time

if __name__ == '__main__':
    model_id = 1001
    device = 0
    start_time = time.time()
    net=model.load_model(model_id,device)
    # #photo
    # img0 = cv2.imread('data/dogandgirl.webp')
    # img1 = data.data_pre_process(img0)
    # output = model.predict(net, img1)
    # output = data.data_post_process(output)
    # img2=data.data_paint(output,img0)
    # cv2.imwrite('data/result.jpg', img2)
    #stream
    rtsp_url = 'rtsp://192.168.2.198:8554/micagent1'
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = data.data_pre_process(frame)
            output = model.predict(net, img)
            output = data.data_post_process(output)
            img = data.data_paint(output, frame)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print('runtime:',end_time-start_time)

