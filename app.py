import os
import io
import base64
import random
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense # Cần cho Dummy Model
from tensorflow.keras.preprocessing.image import img_to_array

# --- CẤU HÌNH ---
app = Flask(__name__)
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
                # THÊM: Buộc ghi dữ liệu xuống đĩa (rất quan trọng trên Windows)
                outfile.flush()
        print(f"✅ Ghép model thành công: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghép model: {e}")
        return False

# Cần glob ở đây
import glob
join_model_parts() 

model = None
is_dummy_model = False # Biến để theo dõi nếu đây là mô hình giả lập
try:
    print(f"🔬 Đang cố gắng tải model từ: {MODEL_PATH}")
    # Compile=False được sử dụng vì chúng ta chỉ đang load để inference (dự đoán)
    model = load_model(MODEL_PATH, compile=False) 
    print("✅ Model thật đã được load.")
    is_dummy_model = False
except Exception as e:
    # Ghi lại lỗi chi tiết khi load model
    print("----------------------------------------------------------")
    print(f"❌ KHÔNG THỂ LOAD MODEL TỪ ĐƯỜNG DẪN: {MODEL_PATH}")
    print(f"Chi tiết lỗi tải model (quan trọng): {e}")
    print("----------------------------------------------------------")
    
    # BƯỚC KHẮC PHỤC: Tạo một mô hình giả lập (Dummy Model)
    print("⚠️ TẠO DUMMY MODEL ĐỂ ỨNG DỤNG TIẾP TỤC CHẠY Ở CHẾ ĐỘ DEMO...")
    try:
        # Cấu hình Dummy Model: Input 224x224x3, trả về 1 xác suất
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
        is_dummy_model = False # Vẫn không có model nào


def preprocess_image(image_file, target_size=(224, 224)):
    """Tiền xử lý hình ảnh cho model Keras."""
    # Mở ảnh từ stream/file object
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    
    # Chuyển đổi sang numpy array và chuẩn hóa (giả định model mong đợi giá trị 0-1)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Thêm dimension batch
    
    # Chuẩn hóa nếu model gốc của bạn sử dụng:
    # VD: img_array = img_array / 255.0 
    
    # TRẢ VỀ: Array sẵn sàng để predict
    return img_array

# --- CÁC ROUTES KHÁC (GIỮ NGUYÊN) ---

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    # Giả lập xác thực (BẠN NÊN THAY THẾ BẰNG FIREBASE HOẶC DB THẬT)
    if username == "user_demo" and password == "123456":
        session['user'] = username
        flash("Đăng nhập thành công!", "success")
        return redirect(url_for("dashboard"))
    else:
        flash("Tên người dùng hoặc mật khẩu không đúng.", "danger")
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.pop('user', None)
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("index"))


# --- ROUTE CHÍNH CÓ CHỈNH SỬA ---

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))
    
    # CHỈNH SỬA: Truyền is_dummy_model
    return render_template("dashboard.html", model_ready=(model is not None), is_dummy=is_dummy_model)


@app.route("/emr_prediction", methods=["GET", "POST"])
def emr_prediction():
    if 'user' not in session:
        flash("Vui lòng đăng nhập.", "danger")
        return redirect(url_for("index"))

    prediction = None
    filename = None
    image_b64 = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash("Không có file nào được chọn.", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("Vui lòng chọn một file hình ảnh.", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Lưu trữ file tạm thời và xử lý
                filename = secure_filename(file.filename)
                
                # Đọc nội dung file để xử lý và tạo Base64
                img_bytes = file.read()
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # --- Xử lý Dự đoán ---
                
                if model is None:
                    flash("Model chưa sẵn sàng. Vui lòng kiểm tra log máy chủ để biết chi tiết.", "danger")
                    # THAY ĐỔI: Truyền trạng thái model_ready
                    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64, model_ready=False)

                # 1. Tiền xử lý ảnh
                x = preprocess_image(io.BytesIO(img_bytes)) 
                
                # 2. Dự đoán với mô hình (thật hoặc giả lập)
                preds = model.predict(x)
                
                # Giả sử mô hình trả về xác suất nhị phân (vd: Nodule)
                score = preds[0][0]
                
                # THÊM: Nếu là dummy model, ghi đè xác suất thành giá trị ngẫu nhiên
                if is_dummy_model:
                     # Ghi đè xác suất ngẫu nhiên để mô phỏng tính demo
                     score = random.uniform(0.4, 0.6) 
                     # THÊM: Cảnh báo flash message
                     flash("⚠️ Kết quả này là từ **Mô hình Giả lập** (DEMO) do không tải được Model thật. Vui lòng kiểm tra file `best_weights_model.keras`.", "warning")

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

    # THAY ĐỔI: Truyền trạng thái model_ready
    return render_template("emr_prediction.html", prediction=prediction, filename=filename, image_b64=image_b64, model_ready=(model is not None))

if __name__ == "__main__":
    app.run(debug=True)
