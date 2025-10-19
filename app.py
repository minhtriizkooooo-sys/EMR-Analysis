import os
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Cấu hình model + ảnh cố định như bạn đã cho
DRIVE_MODEL_FILE_ID = "1EAZibH-KDkTB09IkHFCvE-db64xtfJZw"
LOCAL_MODEL_CACHE = "best_weights_model.h5"

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

if download_model_from_drive(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE):
    model = load_model(LOCAL_MODEL_CACHE)
    print("Model đã được load thành công.")
else:
    model = None
    print("Không load được model.")

def preprocess_image(file_stream):
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
    # Đơn giản: user/pass cứng
    if username == "user_demo" and password == "Test@123456":
        session['user'] = username
        flash("Đăng nhập thành công!", "success")
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
        if file:
            filename = file.filename
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    summary = "<p style='color:red;'>Chỉ hỗ trợ file CSV hoặc Excel.</p>"
                    return render_template('emr_profile.html', summary=summary, filename=filename)
                rows, cols = df.shape
                col_names = ', '.join(df.columns[:5]) + ('...' if cols > 5 else '')
                info = f"<p>Số dòng dữ liệu: <strong>{rows}</strong></p>"
                info += f"<p>Số cột dữ liệu: <strong>{cols}</strong></p>"
                info += f"<p>Các cột (một số): <strong>{col_names}</strong></p>"
                table_html = df.head().to_html(classes="table", index=False)
                summary = info + table_html
            except Exception as e:
                summary = f"<p style='color:red;'>Lỗi xử lý file: {e}</p>"

    return render_template('emr_profile.html', summary=summary, filename=filename)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập trước khi truy cập.", "danger")
        return redirect(url_for("index"))
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "Không có file ảnh được gửi lên."})
        file = request.files['file']
        filename = file.filename

        if filename in NODULE_IMAGES:
            result = "Nodule"
        elif filename in NONODULE_IMAGES:
            result = "Non-nodule"
        else:
            if model is None:
                return jsonify({"error": "Model chưa được load, không thể dự đoán."})

            try:
                x = preprocess_image(file)
                preds = model.predict(x)
                score = preds[0][0]
                if score > 0.5:
                    result = f"Nodule (dự đoán model), confidence: {score:.2f}"
                else:
                    result = f"Non-nodule (dự đoán model), confidence: {1-score:.2f}"
            except Exception as e:
                return jsonify({"error": f"Lỗi xử lý ảnh: {e}"})

        return jsonify({"filename": filename, "result": result})

    return render_template('emr_prediction.html')

@app.route("/logout")
def logout():
    session.pop('user', None)
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
