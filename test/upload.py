from flask import Flask,request,render_template
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_PATH = os.path.join(os.path.dirname(__file__),'images')
@app.route('/upload/',methods=['GET','POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        desc = request.form.get("desc")
        pichead = request.files.get("pichead")
        #filename = secure_filename(pichead.filename) #包装一下 保证文件安全
        #pichead.save(os.path.join(UPLOAD_PATH,pichead.filename)) #可优化
        pichead.save(os.path.join(UPLOAD_PATH,filename)) #已优化
        print(desc)
        return '文件上传成功'

if __name__ == '__main__':
    app.run(debug=True)
