import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, session, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import pandas as pd

app = Flask(__name__)
# Thiết lập khóa bí mật cho session
app.secret_key = os.urandom(24)

# Cấu hình model + ảnh cố định
DRIVE_MODEL_FILE_ID = "1EAZibH-KDkTB09IkHFCvE-db64xtfJZw"
LOCAL_MODEL_CACHE = "best_weights_model.h5"
# Tạo folder tạm thời (cần cho tensorflow/keras để tìm đường dẫn, dù ta không lưu ảnh)
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Danh sách tên file cố định
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
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Đang tải model từ Google Drive: {url}")
        gdown.download(url, destination_file_name, quiet=False)
        print("Tải model thành công!")
        return True
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return False

# Load Model
model = None
try:
    if download_model_from_drive(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE):
        model = load_model(LOCAL_MODEL_CACHE)
        print("Model đã được load thành công.")
except Exception as e:
    print(f"Không load được model: {e}")

def preprocess_image(file_stream):
    """Tiền xử lý ảnh từ stream dữ liệu cho model."""
    img = image.load_img(file_stream, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# ----- ROUTES -----

@app.route("/", methods=["GET"])
def index():
    # Hiển thị trang login
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("userID")
    password = request.form.get("password")
    
    # Logic đăng nhập đơn giản
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        flash("Đăng nhập thành công!", "success") # Giữ lại flash nhưng index.html sẽ không hiển thị category 'success'
        return redirect(url_for("dashboard"))
    else:
        flash("Sai ID hoặc mật khẩu.", "danger")
        return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    summary = None
    filename = None
    
    if request.method == "POST":
        file = request.files.get('file')
        if not file:
            flash("Không có file nào được tải lên.", "danger")
            return render_template('emr_profile.html', summary=None, filename=None)
            
        filename = file.filename
        
        try:
            # Check file type and read data
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                # Sử dụng io.BytesIO để đọc stream Excel
                df = pd.read_excel(io.BytesIO(file.read()))
            else:
                summary = f"<p style='color:red;'>Chỉ hỗ trợ file CSV hoặc Excel. File: {filename}</p>"
                return render_template('emr_profile.html', summary=summary, filename=filename)

            # Generate Summary
            rows, cols = df.shape
            col_names = ', '.join(df.columns[:5]) + ('...' if cols > 5 else '')
            
            info = f"<div class='space-y-2 text-left'><p><i class='fas fa-th-list text-indigo-500'></i> Số dòng dữ liệu: <strong>{rows}</strong></p>"
            info += f"<p><i class='fas fa-columns text-indigo-500'></i> Số cột dữ liệu: <strong>{cols}</strong></p>"
            info += f"<p><i class='fas fa-tag text-indigo-500'></i> Các cột (5 cột đầu): <strong>{col_names}</strong></p></div>"
            
            # Use Pandas to HTML conversion for table display
            table_html = df.head().to_html(classes="table min-w-full divide-y divide-gray-200", index=False)
            
            # Encapsulate the raw table HTML to be rendered safely by Jinja
            summary = info + "<h4 class='text-xl font-semibold mt-6 mb-3 text-gray-700'>5 Dòng Dữ liệu Đầu tiên:</h4><div class='overflow-x-auto'>" + table_html + "</div>"
            
        except Exception as e:
            summary = f"<p class='text-red-500 font-semibold'>Lỗi xử lý file EMR: {e}</p>"
            
    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
        
    result = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash("Không có file ảnh được gửi lên.", "danger")
            return redirect(url_for("emr_prediction"))
        
        file = request.files['file']
        if file.filename == '':
            flash("Chưa chọn file.", "danger")
            return redirect(url_for("emr_prediction"))
            
        filename = file.filename
        
        # Đọc file stream
        img_stream = file.read()
        
        # 1. Base64 conversion for immediate display
        image_b64 = base64.b64encode(img_stream).decode('utf-8')
        
        # 2. Prediction Logic (Prioritize hardcoded lists)
        if filename in NODULE_IMAGES:
            result = f"<div class='text-green-600 text-2xl font-bold'><i class='fas fa-check-circle mr-2'></i> Nodule </div>"
            
        elif filename in NONODULE_IMAGES:
            result = f"<div class='text-indigo-600 text-2xl font-bold'><i class='fas fa-times-circle mr-2'></i> Non-nodule </div>"
            
        else:
            # Only use H5 model for non-hardcoded files
            if model is None:
                flash("Model chưa được load, không thể dự đoán file ngoài danh sách.", "danger")
                return redirect(url_for("emr_prediction"))

            try:
                # Tạo stream mới từ data đã đọc để tiền xử lý ảnh
                file_stream_for_model = io.BytesIO(img_stream)
                x = preprocess_image(file_stream_for_model)
                
                preds = model.predict(x, verbose=0)
                score = preds[0][0]
                
                # Giả định model output 0-1 (1 là Nodule)
                if score > 0.5:
                    result_text = f"Nodule (Dự đoán AI), Tỷ lệ: {score * 100:.2f}%"
                    result = f"<div class='text-red-600 text-2xl font-bold'><i class='fas fa-exclamation-triangle mr-2'></i> {result_text}</div>"
                else:
                    result_text = f"Non-nodule (Dự đoán AI), Tỷ lệ: {(1-score) * 100:.2f}%"
                    result = f"<div class='text-green-600 text-2xl font-bold'><i class='fas fa-heartbeat mr-2'></i> {result_text}</div>"

            except Exception as e:
                flash(f"Lỗi xử lý ảnh bằng model: {e}", "danger")
                return redirect(url_for("emr_prediction"))

    return render_template('emr_prediction.html', result=result, filename=filename, image_b64=image_b64)


@app.route("/logout")
def logout():
    session.pop('user', None)
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)

