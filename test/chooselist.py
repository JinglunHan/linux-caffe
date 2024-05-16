from flask import Flask, render_template, request,redirect

app = Flask(__name__)

# 渲染 HTML 模板
@app.route('/')
def index():
    return render_template('chooselist.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('chooselist.html',file_upload='no file part')
    
    file = request.files['file']
    modelid = request.form['model']
    
    if file.filename == '':
        return render_template('chooselist.html',file_upload='no target file')
    
    # 这里可以对上传的文件进行处理，比如保存到服务器或者进行其他操作
    # 在这个例子中，我们只是返回文件名
    return render_template('chooselist.html',file_upload=format(file.filename),model_load=modelid)
def model():
    deviceid = request.form['device']
    modelid = request.form['model']
    return modelid,deviceid


if __name__ == '__main__':
    app.run(debug=True)