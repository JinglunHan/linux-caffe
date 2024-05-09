from flask import Flask, send_from_directory,render_template,Response

app = Flask(__name__)

@app.route('/')
def index():
    # 返回网页模板，包含视频播放区域
    return render_template('video.html')

@app.route('/stream')
def serve_video():
    file = open('static/move.mp4','rb')
    response = Response(file.read(),mimetype = 'video/mp4')
    response.headers['Content-Disposition'] = 'inline'
    return response

if __name__ == '__main__':
    app.run(host='192.168.2.4', port=5000,debug=True)