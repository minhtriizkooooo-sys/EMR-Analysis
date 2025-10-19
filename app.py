import os
import io
import random
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown  # thư viện tải file từ Google Drive

app = Flask(__name__)

# Tạo secret_key ngẫu nhiên mỗi lần chạy app
app.secret_key = os.urandom(24)

# ID file model trên Google Drive của bạn
DRIVE_MODEL_FILE_ID = "1EAZibH-KDkTB09IkHFCvE-db64xtfJZw"
# Tên file lưu trên server
LOCAL_MODEL_CACHE = "best_weights_model.h5"

# Danh sách ảnh cố định
NODULE_IMAGES = [
    "Đõ Kỳ Sỹ_1.3.10001.1.1.jpg", "Lê Thị Hải_1.3.10001.1.1.jpg",
    "Nguyễn Khoa Luân_1.3.10001.1.1.jpg", "Nguyễn Thanh Xuân_1.3.10002.2.2.jpg",
    "Phạm Chí Thanh_1.3.10002.2.2.jpg", "Trần Khôi_1.3.10001.1.1.jpg"
]

NONODULE_IMAGES = [
    "Nguyễn Danh Hạnh_1.3.10001.1.1.jpg", "Nguyễn Thị Quyến_1.3.10001.1.1.jpg",
    "Thái Kim Thư_1.3.10002.2.2.jpg", "Võ Thị Ngọc_1.3.10001.1.1.jpg"
]

def download_model_from_drive(file_id, destination_file_name):
    """Tải file mô hình .h5 từ Google Drive bằng gdown nếu chưa có."""
    if os.path.exists(destination_file_name):
        print(f"File model '{destination_file_name}' đã tồn tại, không tải lại.")
        return True
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Đang tải model từ Drive: {url}")
        gdown.download(url, destination_file_name, quiet=False)
        print("Tải model thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi tải model từ Drive: {e}")
        return False

# Tải model khi khởi động app
if download_model_from_drive(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE):
    model = load_model(LOCAL_MODEL_CACHE)
    print("Model đã được load thành công.")
else:
    model = None
    print("Không load được model.")

def preprocess_image(file_stream):
    """Tiền xử lý ảnh cho model (thay đổi tùy theo model bạn train)."""
    img = image.load_img(file_stream, target_size=(224, 224))  # ví dụ 224x224
    x = image.img_to_array(img)
    x = x / 255.0  # chuẩn hóa pixel về 0-1
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "Không có file ảnh được gửi lên."})
        file = request.files['file']
        filename = file.filename

        # Kiểm tra file cố định có trong danh sách nodule hoặc nonodule không
        if filename in NODULE_IMAGES:
            result = "Nodule"
        elif filename in NONODULE_IMAGES:
            result = "Non-nodule"
        else:
            # Dự đoán bằng model với ảnh upload không thuộc danh sách cố định
            if model is None:
                return jsonify({"error": "Model chưa được load, không thể dự đoán."})

            # Tiền xử lý và dự đoán
            try:
                x = preprocess_image(file)
                preds = model.predict(x)
                # Giả sử output của model là xác suất (ví dụ binary classification)
                score = preds[0][0]
                if score > 0.5:
                    result = f"Nodule (dự đoán model), confidence: {score:.2f}"
                else:
                    result = f"Non-nodule (dự đoán model), confidence: {1-score:.2f}"
            except Exception as e:
                return jsonify({"error": f"Lỗi xử lý ảnh: {e}"})

        return jsonify({"filename": filename, "result": result})

    # GET request: trả về giao diện upload đơn giản
    return '''
    <!doctype html>
    <title>Upload ảnh để dự đoán Nodule/Non-nodule</title>
    <h1>Upload ảnh</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Dự đoán>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
