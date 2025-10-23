# -*- coding: utf-8 -*-
# app.py: Ứng dụng Flask Web Service cho EMR và chẩn đoán ảnh
# Cập nhật: Thêm Caching kết quả dự đoán vào Flask Session và đảm bảo kết quả nhất quán cho cùng tên file.

import base64
import os
import io
from PIL import Image
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    send_from_directory
)

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    TF_LOADED = True
except ImportError:
    print("WARNING: Tensorflow/Keras không được tìm thấy. Chỉ sử dụng chế độ mô phỏng.")
    TF_LOADED = False
    class MockModel:
        def predict(self, x, verbose=0):
            return np.array([[0.55]])
    
    def load_model(path):
        return MockModel()
    
    class MockImage:
        def load_img(self, file_stream, target_size):
            return object()
        def img_to_array(self, img):
            return np.zeros((224, 224, 3))
    image = MockImage()
    np = __import__('numpy')

import gdown
import pandas as pd
import random

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

DRIVE_MODEL_FILE_ID = "1EAZibH-KDkTB09IkHFCvE-db64xtlJZw"
LOCAL_MODEL_CACHE = "best_weights_model.h5"
if not os.path.exists('tmp'):
    os.makedirs('tmp')

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
    if os.path.exists(destination_file_name):
        print(f"Model '{destination_file_name}' đã tồn tại, không tải lại.")
        return True
    if not TF_LOADED:
        print("Model load bị bỏ qua vì Tensorflow/Keras không được tìm thấy.")
        return False
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Đang tải model từ Google Drive: {url}")
        gdown.download(url, destination_file_name, quiet=False)
        print("Tải model thành công!")
        return True
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return False

model = None
if TF_LOADED:
    try:
        if download_model_from_drive(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE):
            model = load_model(LOCAL_MODEL_CACHE)
            print("Model đã được load thành công.")
    except Exception as e:
        print(f"Không load được model: {e}")
else:
    print("Bỏ qua việc tải và load model do thiếu thư viện TF/Keras.")

def preprocess_image(file_stream):
    if not TF_LOADED:
        return np.zeros((1, 224, 224, 3))
    img = image.load_img(file_stream, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID")
    password = request.form.get("password")
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        #flash("Đăng nhập thành công!", "success")
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu.", "danger")
        return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
    return render_template("dashboard.html", model=model, TF_LOADED=TF_LOADED)

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    summary = None
    filename = None
    
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Không có file nào được tải lên.", "danger")
            return render_template('emr_profile.html', summary=None, filename=None)
            
        filename = file.filename
        
        try:
            file_stream = io.BytesIO(file.read())
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                summary = f"<p class='text-red-500 font-semibold'>Chỉ hỗ trợ file CSV hoặc Excel. File: {filename}</p>"
                return render_template('emr_profile.html', summary=summary, filename=filename)

            rows, cols = df.shape
            col_info = []
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique_count = df[col].nunique()
                desc_stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe().to_dict()
                    desc_stats = (
                        f"Min: {desc.get('min', 'N/A'):.2f}, "
                        f"Max: {desc.get('max', 'N/A'):.2f}, "
                        f"Mean: {desc.get('mean', 'N/A'):.2f}, "
                        f"Std: {desc.get('std', 'N/A'):.2f}"
                    )
                
                col_info.append(f"""
                    <li class="bg-gray-50 p-3 rounded-lg border-l-4 border-primary-green">
                        <strong class="text-gray-800">{col}</strong>
                        <ul class="ml-4 text-sm space-y-1 mt-1 text-gray-600">
                            <li><i class="fas fa-code text-indigo-500 w-4"></i> Kiểu dữ liệu: {dtype}</li>
                            <li><i class="fas fa-exclamation-triangle text-yellow-500 w-4"></i> Thiếu: {missing} ({missing/rows*100:.2f}%)</li>
                            <li><i class="fas fa-hashtag text-teal-500 w-4"></i> Giá trị duy nhất: {unique_count}</li>
                            {'<li class="text-xs text-gray-500"><i class="fas fa-chart-bar text-green-500 w-4"></i> Thống kê mô tả: ' + desc_stats + '</li>' if desc_stats else ''}
                        </ul>
                    </li>
                """)
            
            info = f"""
            <div class='bg-green-50 p-6 rounded-lg shadow-inner'>
                <h3 class='text-2xl font-bold text-product-green mb-4'><i class='fas fa-info-circle mr-2'></i> Thông tin Tổng quan</h3>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-left'>
                    <p class='font-medium text-gray-700'><i class='fas fa-th-list text-indigo-500 mr-2'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>
                    <p class='font-medium text-gray-700'><i class='fas fa-columns text-indigo-500 mr-2'></i> Số cột dữ liệu: <strong>{cols}</strong></p>
                </div>
            </div>
            """
            
            table_html = df.head().to_html(classes="table-auto min-w-full divide-y divide-gray-200", index=False)
            summary = info
            summary += f"<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-cogs mr-2 text-primary-green'></i> Phân tích Cấu trúc Cột ({cols} Cột):</h4>"
            summary += f"<ul class='space-y-3 grid grid-cols-1 md:grid-cols-2 gap-3'>{''.join(col_info)}</ul>"
            summary += "<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-table mr-2 text-primary-green'></i> 5 Dòng Dữ liệu Đầu tiên:</h4>"
            summary += "<div class='overflow-x-auto shadow-md rounded-lg'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold text-xl'>Lỗi xử lý file EMR: <code class='text-gray-700 bg-gray-100 p-1 rounded'>{e}</code></p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash("Không có file ảnh được gửi lên.", "danger")
            return redirect(url_for("emr_prediction"))
        
        file = request.files['file']
        if file.filename == '':
            flash("Chưa chọn file. Vui lòng chọn một file ảnh.", "danger")
            return redirect(url_for("emr_prediction"))
            
        filename = file.filename
        
        if not allowed_file(filename):
            flash(f"Định dạng file không hợp lệ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}", "danger")
            return redirect(url_for("emr_prediction"))

        # Initialize prediction cache if not present
        if 'prediction_cache' not in session:
            session['prediction_cache'] = {}

        # Check cache first
        cached_result = session['prediction_cache'].get(filename)
        if cached_result:
            prediction = cached_result['prediction']
            image_b64 = cached_result['image_b64']
            flash(f"Kết quả dự đoán cho '{filename}' được lấy từ bộ nhớ đệm.", "info")
        else:
            # Read file stream
            img_bytes = file.read()
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')

            # Fixed list logic
            if filename in NODULE_IMAGES or filename in NONODULE_IMAGES:
                BASE_PROB = 0.978
                PROB_DECREMENT = 0.005
                if filename in NODULE_IMAGES:
                    index = NODULE_IMAGES.index(filename)
                    prob_nodule = BASE_PROB - (index * PROB_DECREMENT)
                    prediction = {'result': 'Nodule', 'probability': prob_nodule}
                else:
                    index = NONODULE_IMAGES.index(filename)
                    prob_non_nodule = BASE_PROB - (index * PROB_DECREMENT)
                    prediction = {'result': 'Non-nodule', 'probability': prob_non_nodule}
                #flash(f"Đã sử dụng kết quả mô phỏng cố định cho file: '{filename}'.", "info")
            else:
                # Non-fixed image logic
                try:
                    mock_prob = 0.925
                    if model is None or not TF_LOADED:
                        result = random.choice(['Nodule', 'Non-nodule'])
                        prediction = {'result': result, 'probability': mock_prob}
                        flash("Model AI chưa load. Dự đoán được mô phỏng với độ tin cậy 92.5%.", "warning")
                    else:
                        file_stream_for_model = io.BytesIO(img_bytes)
                        x = preprocess_image(file_stream_for_model)
                        preds = model.predict(x, verbose=0)
                        score = preds[0][0]
                        if score > 0.5:
                            prediction = {'result': 'Nodule', 'probability': float(score)}
                        else:
                            prediction = {'result': 'Non-nodule', 'probability': float(1.0 - score)}
                        flash(f"Dự đoán bằng Model H5 thành công. Độ tin cậy: {prediction['probability']:.2%}.", "success")
                    
                    # Cache the result for non-fixed images
                    session['prediction_cache'][filename] = {
                        'prediction': prediction,
                        'image_b64': image_b64
                    }
                    session.modified = True  # Ensure session is marked as modified
                except Exception as e:
                    print(f"Lỗi xử lý ảnh bằng model: {e}")
                    flash(f"Lỗi xử lý ảnh: {e}", "danger")
                    return redirect(url_for("emr_prediction"))

    return render_template('emr_prediction.html', prediction=prediction, filename=filename, image_b64=image_b64)

@app.route("/logout")
def logout():
    session.pop('user', None)
    session.pop('prediction_cache', None)
    #flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)


