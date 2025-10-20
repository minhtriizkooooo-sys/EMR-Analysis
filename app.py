# -*- coding: utf-8 -*-
# EMR Insight AI - Flask Application

import os
import io
import glob
import base64
import random
import numpy as np
import pandas as pd
from PIL import Image
from flask import (
    Flask, flash, redirect, render_template,
    request, session, url_for
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Model Config ---
MODEL_DIR = "models"
MODEL_FILENAME = "best_weights_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ghép các part thành model nếu chưa có
def join_model_parts():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model đã tồn tại: {MODEL_PATH}")
        return

    model_parts = sorted(glob.glob(os.path.join(MODEL_DIR, "best_weights_model.keras.part*")))
    if not model_parts:
        print("❌ Không tìm thấy phần nào của model.")
        return

    print(f"🔧 Đang ghép {len(model_parts)} phần model...")
    try:
        with open(MODEL_PATH, "wb") as outfile:
            for part in model_parts:
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
        print("✅ Ghép model thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")

# Gọi hàm ghép part
join_model_parts()

# Load model thật
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model thật đã được load.")
except Exception as e:
    print(f"❌ Không thể load model thật: {e}")
    model = None

# --- Helper ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_stream):
    img = image.load_img(file_stream, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
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
    return render_template("dashboard.html")

@app.route("/emr_profile", methods=["GET", "POST"])
def emr_profile():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))

    summary = None
    filename = None

    if request.method == "POST":
        file = request.files.get('file')
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

        except Exception as e:
            summary = f"<p>Lỗi xử lý file: {e}</p>"

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
            flash("Vui lòng chọn file ảnh hợp lệ.", "danger")
            return redirect(url_for("emr_prediction"))

        filename = file.filename
        try:
            img_bytes = file.read()
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')

            if model is None:
                flash("Model chưa sẵn sàng.", "danger")
                return redirect(url_for("emr_prediction"))

            x = preprocess_image(io.BytesIO(img_bytes))
            preds = model.predict(x)
            score = preds[0][0]

            label = "Nodule" if score > 0.5 else "Non-nodule"
            probability = float(score if score > 0.5 else 1 - score)
            prediction = {"result": label, "probability": probability}

            flash(f"Kết quả: {label} - {probability:.2%}", "success")
        except Exception as e:
            flash(f"Lỗi xử lý ảnh: {e}", "danger")

    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64)

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "success")
    return redirect(url_for("index"))

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
