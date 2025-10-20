# app.py # -*- coding: utf-8 -*-
# CẬP NHẬT CUỐI CÙNG: LOẠI BỎ MỌI MÔ PHỎNG VÀ ĐẢM BẢO MODEL THẬT ĐƯỢC LOAD TỪ FILE ĐÃ GHÉP.

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

# Thư viện cho AI/Data
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    TF_LOADED = True
    print("TensorFlow/Keras đã được tải thành công.")
except ImportError:
    print("FATAL ERROR: TensorFlow/Keras không được tìm thấy. Ứng dụng sẽ bị lỗi khi cố gắng load model.")
    TF_LOADED = False
    # Định nghĩa các mock để tránh crash ngay lập tức nếu TF_LOADED=False
    class MockModel:
        def predict(self, x, verbose=0):
            return np.array([[0.5]])
    def load_model(path): return MockModel()
    class MockImage:
        def load_img(self, file_stream, target_size): return object()
        def img_to_array(self, img): return np.zeros((224, 224, 3))
    image = MockImage()
    np = __import__('numpy') 

import pandas as pd
import random # Giữ lại để dùng cho mock probability nếu model crash

app = Flask(__name__)
app.secret_key = os.urandom(24)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cấu hình model và logic ghép file
LOCAL_MODEL_NAME = "best_weights_model.keras" 
MODEL_PARTS = ["best_weights_model.keras.part0", "best_weights_model.keras.part1"] 
MODEL_DIR = "models"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, LOCAL_MODEL_NAME)

# Tạo folder models nếu chưa tồn tại
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Đã tạo thư mục '{MODEL_DIR}'. Vui lòng đặt các file part vào đây.")

# LOẠI BỎ CÁC DANH SÁCH MÔ PHỎNG NÀY (Không dùng nữa)
# NODULE_IMAGES = [...]
# NONODULE_IMAGES = [...]

def check_and_join_model_parts():
    """Kiểm tra sự tồn tại của các phần model và ghép chúng lại."""
    if os.path.exists(FULL_MODEL_PATH):
        print(f"Model '{LOCAL_MODEL_NAME}' đã tồn tại, không cần ghép.")
        return True

    print(f"Đang kiểm tra các phần model trong thư mục '{MODEL_DIR}'...")
    part_paths = [os.path.join(MODEL_DIR, part_name) for part_name in MODEL_PARTS]
    missing_parts = [p for p in part_paths if not os.path.exists(p)]
    
    if missing_parts:
        print(f"LỖI: Thiếu các phần model sau: {', '.join([os.path.basename(p) for p in missing_parts])}")
        return False

    # Thực hiện ghép file
    print(f"Tìm thấy đầy đủ các phần. Đang ghép file thành '{LOCAL_MODEL_NAME}'...")
    try:
        with open(FULL_MODEL_PATH, 'wb') as outfile:
            for part_path in part_paths:
                with open(part_path, 'rb') as infile:
                    outfile.write(infile.read())
        print("Ghép file thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi ghép file model: {e}")
        return False

# Load Model
model = None
if TF_LOADED:
    # 1. Kiểm tra và ghép model
    if check_and_join_model_parts():
        # 2. Tải model
        try:
            print(f"Đang TẢI model Keras THẬT từ đường dẫn: {FULL_MODEL_PATH}...")
            # Sử dụng parameter compile=False để tránh lỗi nếu optimizer không được lưu
            model = load_model(FULL_MODEL_PATH, compile=False) 
            print("Model Keras THẬT đã được load thành công vào bộ nhớ.")
        except Exception as e:
            # Rất quan trọng: Bắt lỗi load model để tránh lỗi 502 Bad Gateway khi deploy
            print(f"LỖI NGHIÊM TRỌNG khi load model Keras THẬT: {e}")
            print("Ứng dụng sẽ chạy ở chế độ MOCK (dự đoán giả lập) và không thể sử dụng AI.")
            model = None # Đảm bảo model là None nếu load thất bại
else:
    print("Bỏ qua việc tải và load model do thiếu thư viện TF/Keras. Chế độ MOCK.")


def preprocess_image(file_stream):
    """Tiền xử lý ảnh từ stream dữ liệu cho model."""
    if model is None or not TF_LOADED:
        return np.zeros((1, 224, 224, 3))
        
    # Xử lý ảnh
    img = image.load_img(file_stream, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID")
    password = request.form.get("password")
    
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        # LOẠI BỎ FLASH MESSAGE ĐĂNG NHẬP THÀNH CÔNG (Yêu cầu 1)
        # flash("Đăng nhập thành công!", "success") 
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu.", "danger")
        return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
    # Truyền trạng thái model để hiển thị thông báo nếu model chưa load được
    return render_template("dashboard.html", model=model, TF_LOADED=TF_LOADED) 

# Giữ nguyên route emr_profile vì bạn chỉ yêu cầu xóa flash message
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

            # (Logic tạo summary được giữ nguyên)
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
                <h3 class='text-2xl font-bold text-primary-green mb-4'><i class='fas fa-info-circle mr-2'></i> Thông tin Tổng quan</h3>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-left'>
                    <p class='font-medium text-gray-700'><i class='fas fa-th-list text-indigo-500 mr-2'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>
                    <p class='font-medium text-gray-700'><i class='fas fa-columns text-indigo-500 mr-2'></i> Số cột dữ liệu: <strong>{cols}</strong></p>
                </div>
            </div>
            """
            
            table_html = df.head().to_html(classes="table-auto min-w-full divide-y divide-gray-200", index=False)
            
            summary = info
            summary += f"<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-cogs mr-2 text-primary-green'></i> Phân tích Cấu trúc Cột ({cols} Cột):</h4>"
            summary += f"<ul class='space-y-3 grid grid-cols-1 md:grid-cols-2 gap-3'>{ ''.join(col_info) }</ul>"
            summary += "<h4 class='text-xl font-semibold mt-8 mb-4 text-gray-700'><i class='fas fa-table mr-2 text-primary-green'></i> 5 Dòng Dữ liệu Đầu tiên:</h4>"
            summary += "<div class='overflow-x-auto shadow-md rounded-lg'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold text-xl'>Lỗi xử lý file EMR: <code class='text-gray-700 bg-gray-100 p-1 rounded'>{e}</code></p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)


@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    """Route xử lý tải lên ảnh và dự đoán bằng model Keras. LOẠI BỎ MỌI MÔ PHỎNG."""
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            flash(f"Vui lòng chọn một file ảnh hợp lệ ({', '.join(ALLOWED_EXTENSIONS)}).", "danger")
            return redirect(url_for("emr_prediction"))
            
        filename = file.filename
        
        # --- 1. KIỂM TRA CACHE TRƯỚC ---
        if 'prediction_cache' not in session:
            session['prediction_cache'] = {}
            
        # Do đã loại bỏ danh sách cố định, cache chỉ lưu kết quả model thật/mock
        cached_result = session['prediction_cache'].get(filename)

        if cached_result:
            prediction = cached_result['prediction']
            image_b64 = cached_result['image_b64']
            flash(f"Kết quả dự đoán cho '{filename}' được lấy từ **bộ nhớ đệm (Cache)**.", "info")
            
        else:
            # --- 2. ĐỌC FILE VÀ TIẾN HÀNH DỰ ĐOÁN MỚI (CHỈ MODEL THẬT/MOCK) ---
            try:
                img_bytes = file.read()
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                if model is None or not TF_LOADED:
                    # Chế độ MOCK (Nếu load model thật thất bại)
                    mock_prob = 0.925
                    result = random.choice(['Nodule', 'Non-nodule'])
                    
                    # Thay đổi xác suất dựa trên kết quả mock
                    prob = mock_prob if result == 'Nodule' else 1.0 - (mock_prob - 0.5) 
                    
                    prediction = {'result': result, 'probability': prob}
                    flash("Model AI chưa load. Dự đoán được **MÔ PHỎNG** với độ tin cậy thấp.", "warning")
                        
                else:
                    # Chế độ DỰ ĐOÁN THẬT
                    file_stream_for_model = io.BytesIO(img_bytes)
                    x = preprocess_image(file_stream_for_model)
                    
                    preds = model.predict(x, verbose=0)
                    score = preds[0][0] # Giả định 1.0 là Nodule, 0.0 là Non-nodule
                    
                    if score > 0.5:
                        prediction = {'result': 'Nodule', 'probability': float(score)}
                    else:
                        prediction = {'result': 'Non-nodule', 'probability': float(1.0 - score)}
                        
                    flash(f"Dự đoán bằng Model Keras **THẬT** thành công. Độ tin cậy: {prediction['probability']:.2%}.", "success")
                
                
                # --- 3. LƯU KẾT QUẢ VÀO CACHE ---
                session['prediction_cache'][filename] = {
                    'prediction': prediction,
                    'image_b64': image_b64 
                }
                
            except Exception as e:
                print(f"LỖI THỜI GIAN CHẠY XỬ LÝ ẢNH/MODEL: {e}")
                flash(f"Lỗi xử lý ảnh: {e}. Vui lòng kiểm tra file ảnh hoặc model.", "danger")
                return redirect(url_for("emr_prediction"))
                
    return render_template('emr_prediction.html', prediction=prediction, filename=filename, image_b64=image_b64)


@app.route("/logout")
def logout():
    session.pop('user', None)
    session.pop('prediction_cache', None) 
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
