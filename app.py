import os
import io
import base64
import random
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
# Imports for Keras/Tensorflow model (needed for model loading/dummy creation)
from tensorflow.keras.models import load_model, Sequential 
from tensorflow.keras.layers import Input, Flatten, Dense 
from tensorflow.keras.preprocessing.image import img_to_array
import glob 

# --- CẤU HÌNH ---
app = Flask(__name__)
# Đảm bảo secret_key được đặt để sử dụng session và flash messages
app.secret_key = "secret_key_emr_insight_ai" 
UPLOAD_FOLDER = 'static/uploads'
MODEL_DIR = 'models'
MODEL_FILENAME = 'best_weights_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- KHỞI TẠO VÀ LOAD MODEL ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def join_model_parts():
    """Kiểm tra và ghép các phần model đã chia nhỏ thành một file duy nhất."""
    model_parts_pattern = os.path.join(MODEL_DIR, f"{MODEL_FILENAME}.part*")
    model_parts = sorted(glob.glob(model_parts_pattern))

    if not model_parts:
        print(f"❌ Không tìm thấy phần nào của model tại: {model_parts_pattern}")
        return False

    print(f"🔧 Đang ghép {len(model_parts)} phần model...")
    try:
        with open(MODEL_PATH, "wb") as outfile:
            for part in model_parts:
                print(f"🧩 Ghép: {part}")
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
                # Buộc ghi dữ liệu xuống đĩa 
                outfile.flush()
        print(f"✅ Ghép model thành công: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

join_model_parts() 

model = None
is_dummy_model = False # Biến để theo dõi nếu đây là mô hình giả lập
try:
    print(f"🔬 Đang cố gắng tải model từ: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False) 
    print("✅ Model thật đã được load.")
    is_dummy_model = False
except Exception as e:
    print("----------------------------------------------------------")
    print(f"❌ KHÔNG THỂ LOAD MODEL TỪ ĐƯỜNG DẪN: {MODEL_PATH}")
    print(f"Chi tiết lỗi tải model: {e}")
    print("----------------------------------------------------------")
    
    # BƯỚC KHẮC PHỤC: Tạo một mô hình giả lập (Dummy Model)
    print("⚠️ TẠO DUMMY MODEL ĐỂ ỨNG DỤNG TIẾP TỤC CHẠY Ở CHẾ ĐỘ DEMO...")
    try:
        model = Sequential([
            Input(shape=(224, 224, 3)),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        print("✅ DUMMY Model đã được tạo thành công.")
        is_dummy_model = True
    except Exception as dummy_e:
        print(f"❌ Lỗi khi tạo Dummy Model: {dummy_e}")
        model = None
        is_dummy_model = False


def preprocess_image(image_file, target_size=(224, 224)):
    """Tiền xử lý hình ảnh cho model Keras."""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Thêm dimension batch
    # Tùy chọn: img_array = img_array / 255.0  # Chuẩn hóa nếu cần
    return img_array

# --- CÁC ROUTES CHÍNH XÁC THEO YÊU CẦU ---

@app.route("/")
@app.route("/index")
def index():
    """Hiển thị trang đăng nhập."""
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    """Xử lý đăng nhập và điều hướng đến dashboard."""
    # SỬ DỤNG ĐÚNG tên trường (name="userID") từ form HTML
    username = request.form.get("userID", "").strip() 
    password = request.form.get("password", "").strip()

    # Giả lập xác thực: user_demo / Test@123456
    if userID == "user_demo" and password == "Test@123456":
        session['user'] = username
        flash("Đăng nhập thành công! Chào mừng đến với Dashboard.", "success")
        # Điều hướng đến dashboard
        return redirect(url_for("dashboard"))
    else:
        flash("Tên người dùng hoặc mật khẩu không đúng.", "danger")
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    """Xử lý đăng xuất."""
    session.pop('user', None)
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    """Hiển thị trang dashboard (điều hướng chính) với các lựa chọn EMR Profile và EMR Prediction."""
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))
    
    # Truyền trạng thái model để hiển thị thông báo trên dashboard
    return render_template("dashboard.html", model_ready=(model is not None), is_dummy=is_dummy_model)

@app.route("/emr_profile")
def emr_profile():
    """Hiển thị trang Hồ sơ Bệnh án EMR."""
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))
    
    # Truyền dữ liệu profile giả lập (nếu cần)
    dummy_profile = {
        "name": "Bệnh nhân Demo",
        "dob": "15/05/1980",
        "gender": "Nam",
        "last_scan": "01/01/2025",
        "diagnosis": "Đang cập nhật...",
        "history": [
            {"date": "2024-05-10", "note": "Kiểm tra định kỳ."},
            {"date": "2023-11-20", "note": "Phát hiện vết mờ nhỏ, theo dõi."},
        ]
    }
    return render_template("emr_profile.html", profile=dummy_profile)

@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    """Hiển thị trang phân tích và xử lý dự đoán EMR."""
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))

    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if 'file' not in request.files or request.files['file'].filename == '':
            flash("Vui lòng chọn một file hình ảnh.", "danger")
            return redirect(request.url)

        file = request.files['file']

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                
                # Đọc nội dung file để xử lý và tạo Base64
                img_bytes = file.read()
                # Base64 cho hình ảnh hiển thị lại trên trang kết quả
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # --- Xử lý Dự đoán ---
                
                if model is None:
                    flash("Model chưa sẵn sàng. Vui lòng kiểm tra log máy chủ.", "danger")
                    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64, model_ready=False)

                x = preprocess_image(io.BytesIO(img_bytes)) 
                
                preds = model.predict(x)
                score = preds[0][0]
                
                if is_dummy_model:
                     # Ghi đè xác suất ngẫu nhiên để mô phỏng tính demo
                     score = random.uniform(0.4, 0.6) 
                     flash("⚠️ Kết quả này là từ **Mô hình Giả lập (DEMO)**. Vui lòng kiểm tra file `best_weights_model.keras`.", "warning")

                # 3. Phân loại và định dạng kết quả
                THRESHOLD = 0.5
                if score >= THRESHOLD:
                    label = "Nodule (U/Nốt)"
                    probability = score
                    advice = "Khuyến nghị: Chụp ảnh thêm hoặc tham khảo ý kiến chuyên gia để xác nhận."
                else:
                    label = "Non-nodule (Không phải U)"
                    probability = 1.0 - score
                    advice = "Khuyến nghị: Theo dõi định kỳ, kết quả dự đoán là lành tính."

                prediction = {
                    "label": label,
                    "probability": f"{probability:.2%}",
                    "advice": advice
                }
                
                flash(f"Dự đoán hoàn tất: {label} - {probability:.2%}", "success")
                
            except Exception as e:
                print(f"❌ Lỗi trong quá trình dự đoán: {e}")
                flash(f"Lỗi hệ thống trong quá trình dự đoán: {e}", "danger")
                
        else:
            flash("Định dạng file không hợp lệ. Chỉ chấp nhận JPG, PNG, GIF, BMP.", "danger")

    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64, model_ready=(model is not None))

if __name__ == "__main__":
    app.run(debug=True)
