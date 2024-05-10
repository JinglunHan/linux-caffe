from flask import Flask, render_template, request

app = Flask(__name__)

# 渲染 HTML 模板
@app.route('/')
def index():
    return render_template('chooselist.html')

# 处理选项的 POST 请求
@app.route('/process_selection', methods=['POST'])
def process_selection():
    selected_option = request.form['model']
    return f"你选择了：{selected_option}"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    # 这里可以对上传的文件进行处理，比如保存到服务器或者进行其他操作
    # 在这个例子中，我们只是返回文件名
    return 'Uploaded file: {}'.format(file.filename)

if __name__ == '__main__':
    app.run(debug=True)