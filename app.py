from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
import os
import secrets
import shutil
import pandas as pd
import numpy as np
# Dùng try-except để đảm bảo load_model không làm crash app nếu TF/Keras lỗi
try:
    from keras.models import load_model 
except ImportError:
    print("❌ Lỗi: Keras/Tensorflow chưa được cài đặt hoặc import.")
    load_model = None 

from PIL import Image
from werkzeug.utils import secure_filename

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'} 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FOLDER = 'models'
# Sử dụng lại tên model đã ghép
MERGED_MODEL_PATH = os.path.join(MODEL_FOLDER, 'best_weights_model_merged.keras') 

# =============================
# Hàm tiện ích
# =============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# =============================
# Ghép các phần model keras (Sử dụng tên file gốc: best_weights_model.keras.001 - .004)
# =============================
def merge_model_files():
    # SỬ DỤNG LẠI TÊN FILE ZIPPED BAN ĐẦU
    parts = [
        os.path.join(MODEL_FOLDER, f"best_weights_model.keras.{i:03d}")
        for i in range(1, 5)
    ]
    
    missing_files = [p for p in parts if not os.path.exists(p)]
    
    if missing_files:
        print("==================================================")
        print("⚠️ THIẾU FILE MODEL QUAN TRỌNG TRONG MÔI TRƯỜNG RENDER!")
        print(f"Thư mục hiện tại: {os.getcwd()}")
        print(f"Các file model CẦN có: {parts}")
        print(f"Các file model BỊ THIẾU: {missing_files}")
        print("⚠️ Vui lòng đảm bảo các file này đã được commit vào thư mục 'models'.")
        print("==================================================")
        return None
    
    if os.path.exists(MERGED_MODEL_PATH):
        print("✅ Model đã được ghép. Bỏ qua bước ghép.")
        return MERGED_MODEL_PATH

    print("🔧 Ghép model...")
    try:
        with open(MERGED_MODEL_PATH, "wb") as merged:
            for part in parts:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, merged)
        print("✅ Đã ghép xong model.")
        return MERGED_MODEL_PATH
    except Exception as e:
        print(f"❌ Lỗi khi ghép model (Có thể do IO/Permission): {e}")
        return None

# =============================
# Load model khi khởi động
# =============================
model = None
if load_model: 
    MODEL_PATH = merge_model_files()
    if MODEL_PATH:
        try:
            model = load_model(MODEL_PATH) 
            print("✅ Model y tế đã load thành công.")
        except Exception as e:
            print("❌ Lỗi khi load model:", e)
            print("❌ Lỗi này có thể do xung đột phiên bản Keras/Tensorflow. Hãy kiểm tra logs build.")
            model = None
else:
    print("❌ Model không thể load do Keras/Tensorflow không được import.")
    

# =============================
# Trang chủ (Đăng nhập)
# =============================
@app.route('/')
def index():
    return render_template('index.html')

# =============================
# Xử lý đăng nhập
# =============================
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('userID')
    password = request.form.get('password')

    if username == 'user_demo' and password == 'Test@123456':
        session['logged_in'] = True
        session['username'] = username
        flash('Đăng nhập thành công!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Sai tài khoản hoặc mật khẩu!', 'danger')
        return redirect(url_for('index'))

# =============================
# Trang Dashboard
# =============================
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập Dashboard.', 'warning')
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session.get('username'))

# =============================
# Trang phân tích hồ sơ EMR
# =============================
@app.route('/emr_profile')
def emr_profile():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    return render_template('emr_profile.html')

# =============================
# Upload & phân tích hồ sơ EMR
# =============================
@app.route('/upload_emr', methods=['POST'])
def upload_emr():
    if not session.get('logged_in'):
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('Không có file nào được tải lên.', 'warning')
        return redirect(url_for('emr_profile'))

    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn file hợp lệ.', 'warning')
        return redirect(url_for('emr_profile'))
    
    # Kiểm tra phần mở rộng file
    if not allowed_file(file.filename):
        flash('Định dạng file không được hỗ trợ!', 'danger')
        return redirect(url_for('emr_profile'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
        summary = df.describe(include='all').to_html(classes='table table-bordered table-sm')
        flash('✅ Phân tích hồ sơ EMR thành công!', 'success')
    except Exception as e:
        summary = f"Lỗi khi đọc file: {e}"
        flash(f'❌ Lỗi khi phân tích hồ sơ: {e}', 'danger')

    return render_template('emr_profile.html', summary=summary, filename=file.filename)

# =============================
# Trang phân tích ảnh y tế
# =============================
@app.route('/emr_prediction')
def emr_prediction():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    return render_template('emr_prediction.html')

# =============================
# Upload ảnh y tế & Dự đoán
# =============================
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if not session.get('logged_in'):
        return redirect(url_for('index'))

    if 'image' not in request.files:
        flash('Không có ảnh nào được tải lên.', 'warning')
        return redirect(url_for('emr_prediction'))

    file = request.files['image']
    if file.filename == '':
        flash('Chưa chọn ảnh hợp lệ.', 'warning')
        return redirect(url_for('emr_prediction'))
    
    if file.filename.rsplit('.', 1)[1].lower() not in ['jpg', 'jpeg', 'png']:
        flash('Vui lòng chỉ tải lên file ảnh (jpg, jpeg, png).', 'danger')
        return redirect(url_for('emr_prediction'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    
    prediction_result = None
    
    if model is None:
        flash('❌ Hệ thống AI chưa được tải. Vui lòng kiểm tra logs để xem model bị thiếu file hay lỗi import thư viện.', 'danger')
        return render_template('emr_prediction.html', result=None) 
    
    try:
        # Chuẩn bị ảnh cho model (224x224, RGB, chuẩn hóa)
        img = Image.open(filepath).convert('RGB').resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Thực hiện dự đoán
        probability = float(model.predict(arr)[0][0]) 
        percent = probability * 100
        
        # Định dạng kết quả dự đoán thành chuỗi HTML
        if probability >= 0.5:
            label = "UNG THƯ/BỆNH LÝ NGHIÊM TRỌNG"
            style = "color: red; font-weight: bold; font-size: 20px;"
        else:
            label = "BÌNH THƯỜNG/KHÔNG PHÁT HIỆN BỆNH LÝ"
            style = "color: green; font-weight: bold; font-size: 20px;"
            
        prediction_result = f"""
            <p><strong>Dự đoán AI:</strong> <span style="{style}">{label}</span></p>
            <p><strong>Xác suất dự đoán:</strong> <span style="font-size: 18px;">{percent:.2f}%</span></p>
            <p class="text-muted">*(Dự đoán dựa trên mô hình CNN/LSTM y tế)</p>
        """
        flash('✅ Dự đoán ảnh y tế thành công!', 'success')
        
    except Exception as e:
        print("❌ Lỗi khi dự đoán:", e)
        flash(f'❌ Lỗi xử lý ảnh và dự đoán: {e}', 'danger')
        prediction_result = None 

    return render_template(
        'emr_prediction.html', 
        image_name=file.filename, 
        result=prediction_result 
    )

# =============================
# Đăng xuất
# =============================
@app.route('/logout')
def logout():
    session.clear()
    flash('Bạn đã đăng xuất.', 'info')
    return redirect(url_for('index'))

# =============================
# Serve file upload
# =============================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# =============================
# Chạy app
# =============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
