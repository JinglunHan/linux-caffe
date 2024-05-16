import subprocess

ffmpeg_path = '/home/roota/workstation/opencv/ffmpeg/bin/'
video_path = '/home/roota/workstation/onnx2caffe/linux-caffe/static/results_video/processed_move.mp4'
result_path = '/home/roota/workstation/onnx2caffe/linux-caffe/static/results_video/result_move.mp4'
ffmpeg_command = '{}ffmpeg -i {} -c:v libx264 -c:a aac -strict -2 {}'.format(ffmpeg_path, video_path, result_path)

# 执行FFmpeg命令
process = subprocess.Popen(ffmpeg_command, shell=True)
process.wait() 