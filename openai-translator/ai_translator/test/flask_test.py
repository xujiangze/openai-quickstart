from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'  # 设置上传文件的存储目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'file' not in request.files:
    #     return '没有文件部分', 400
    if 'files' not in request.files:
        return jsonify({'message': 'No files part'}), 400

    files = request.files.getlist('files')
    filenames = []
    for file in files:
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if file:
            # 可以在这里保存文件
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            filenames.append(file.filename)
    return jsonify({'message': 'Files successfully uploaded', 'filenames': filenames}), 200


if __name__ == '__main__':
    # 创建上传文件的存储目录
    if os.path.exists(os.path.dirname(app.config['UPLOAD_FOLDER'])):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 启动服务
    app.run(debug=True)

    # 测试用例
    # curl -X POST -F 'files=@/path/to/your/file1.txt' -F 'files=@/path/to/your/file2.txt' http://localhost:5000/upload
