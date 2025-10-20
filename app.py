# -*- coding: utf-8 -*-
# EMR Insight AI - Flask Application

import os
import io
import glob
import base64
import random
import re
import numpy as np
import pandas as pd
from PIL import Image
from flask import (
    Flask, flash, redirect, render_template,
    request, session, url_for, jsonify
)

# Import các thư viện cần thiết cho mô hình
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Flask Setup ---
app = Flask(__name__)
# KHUYẾN NGHỊ: Đổi secret key khi deploy thật
app.secret_key = os.urandom(24) 

# --- Model Config ---
MODEL_DIR = "models"
MODEL_FILENAME = "best_weights_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
IMAGE_SIZE = (224, 224) # Kích thước đầu vào chuẩn của mô hình (thường là 224x224 cho nhiều CNN)

# --- Chuẩn bị Thư mục và Ghép các phần model ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def join_model_parts():
    """Ghép các phần model đã chia nhỏ thành file model hoàn chỉnh."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model đã tồn tại: {MODEL_PATH}")
        return

    part_pattern = os.path.join(MODEL_DIR, "best_weights_model.keras.part*")
    model_parts = glob.glob(part_pattern)

    def extract_part_number(path):
        match = re.search(r'\.part(\d+)', path)
        return int(match.group(1)) if match else -1

    model_parts = sorted(model_parts, key=extract_part_number)

    if not model_parts:
        print("❌ Không tìm thấy phần nào của model.")
        return

    print(f"🔧 Đang ghép {len(model_parts)} phần model...")
    try:
        with open(MODEL_PATH, "wb") as outfile:
            for part in model_parts:
                print(f"🧩 Ghép: {part}")
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
        print("✅ Ghép model thành công: ", MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")

# Gọi ghép model nếu cần
join_model_parts()

# --- Load model ---
model = None
try:
    # Compile=False được sử dụng vì chúng ta chỉ đang load để inference (dự đoán)
    model = load_model(MODEL_PATH, compile=False) 
    print("✅ Model thật đã được load.")
except Exception as e:
    # Ghi lại lỗi chi tiết khi load model
    print(f"❌ KHÔNG THỂ LOAD MODEL TỪ ĐƯỜNG DẪN: {MODEL_PATH}")
    print(f"Chi tiết lỗi: {e}")
    model = None

# --- Helper Functions ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Kiểm tra định dạng file ảnh hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_stream):
    """Tiền xử lý ảnh cho mô hình dự đoán Nodule Detection (Resize, Normalize)."""
    # load_img cần một file path hoặc stream
    img = image.load_img(file_stream, target_size=IMAGE_SIZE)
    # Chuyển đổi sang array
    x = image.img_to_array(img)
    # Chuẩn hóa (normalize) về 0-1
    x = x / 255.0
    # Thêm chiều batch (batch dimension)
    x = np.expand_dims(x, axis=0)
    return x

# --- Routes ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID")
    password = request.form.get("password")
    # Giữ nguyên logic đăng nhập demo
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu.", "danger") 
        return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))
    # Truyền trạng thái model để template có thể hiển thị cảnh báo
    return render_template("dashboard.html", model_ready=(model is not None))

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))

    summary = None
    filename = None

    if request.method == "POST":
        file = request.files.get('file')
        # ... (Giữ nguyên logic xử lý file CSV/Excel) ...
        if not file or file.filename == '':
            flash("Không có file nào được chọn.", "danger")
            return render_template('emr_profile.html', summary=None)

        filename = file.filename
        try:
            file_stream = io.BytesIO(file.read())
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_stream)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_stream)
            else:
                flash("Chỉ hỗ trợ file CSV hoặc Excel.", "danger")
                return render_template('emr_profile.html', summary=None)

            rows, cols = df.shape
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                unique = df[col].nunique()
                desc_stats = ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe()
                    desc_stats = f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}"
                col_info.append(f"""
                    <li><strong>{col}</strong> - Kiểu: {dtype}, Thiếu: {missing}, Duy nhất: {unique}
                    {'<br> Thống kê: ' + desc_stats if desc_stats else ''}</li>
                """)

            info = f"<p><strong>Số dòng:</strong> {rows} | <strong>Số cột:</strong> {cols}</p>"
            table_html = df.head().to_html(classes="table table-striped", index=False)
            summary = info + "<ul>" + "".join(col_info) + "</ul>" + "<h4>5 dòng đầu tiên:</h4>" + table_html
            flash(f"Phân tích file {filename} thành công!", "success") # Thêm flash thành công

        except Exception as e:
            # Ghi log lỗi và thông báo cho người dùng
            print(f"Lỗi khi phân tích EMR: {e}") 
            summary = f"<p class='text-red-500'>Lỗi xử lý file: {e}</p>"
            flash("Có lỗi xảy ra trong quá trình phân tích file.", "danger")

    return render_template("emr_profile.html", summary=summary, filename=filename)


@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))

    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash("Vui lòng chọn file ảnh hợp lệ (JPG, PNG, GIF, BMP).", "danger")
            return redirect(url_for("emr_prediction"))

        filename = file.filename
        try:
            img_bytes = file.read()
            # Mã hóa ảnh để hiển thị trên web
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')

            if model is None:
                # Thông báo model chưa sẵn sàng
                flash("Model chưa sẵn sàng. Vui lòng kiểm tra log máy chủ để biết chi tiết.", "danger")
                return render_template("emr_prediction.html", model_ready=False)

            # 1. Tiền xử lý ảnh
            x = preprocess_image(io.BytesIO(img_bytes)) 
            
            # 2. Dự đoán với mô hình thật
            preds = model.predict(x)
            
            # Giả sử mô hình trả về xác suất nhị phân (xác suất của lớp '1' - Nodule)
            # preds[0][0] là xác suất của lớp Nodule
            score = preds[0][0] 

            # 3. Phân loại và định dạng kết quả
            THRESHOLD = 0.5
            
            if score >= THRESHOLD:
                # Nếu score >= 0.5, dự đoán là Nodule
                label = "Nodule"
                probability = score
            else:
                # Nếu score < 0.5, dự đoán là Non-nodule
                label = "Non-nodule"
                probability = 1.0 - score 
            
            prediction = {
                "result": label, 
                "probability": probability
            }
            
            # Hiển thị thông báo thành công
            flash(f"Dự đoán hoàn tất: {label} - {probability:.2%}", "success")
            
        except Exception as e:
            # Ghi log lỗi chi tiết khi xảy ra lỗi trong quá trình dự đoán
            print(f"LỖI XỬ LÝ ẢNH HOẶC DỰ ĐOÁN: {e}")
            flash(f"Lỗi xử lý ảnh hoặc dự đoán: {e}", "danger")

    # Truyền trạng thái model để template có thể kiểm tra trạng thái
    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64, model_ready=(model is not None))

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
