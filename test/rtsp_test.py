import cv2

def pull_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        # 在这里可以对帧进行处理，例如显示到窗口或保存到文件
        cv2.imshow('RTSP Stream', frame)
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 拉取单个 RTSP 流的示例
rtsp_url = "rtsp://192.168.2.198:8554/micagent1"
pull_rtsp_stream(rtsp_url)