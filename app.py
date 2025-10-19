from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
import os
import secrets
import pandas as pd
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
# Dùng try-except để đảm bảo load_model không làm crash app nếu TF/Keras lỗi
try:
    from keras.models import load_model 
    import gdown # Thư viện tải file từ Drive
except ImportError as e:
    print(f"❌ Lỗi: Keras/Tensorflow/gdown chưa được cài đặt hoặc import. Chi tiết: {e}")
    load_model = None 
    gdown = None

# =============================
# Cấu hình Flask
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'} 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FOLDER = 'models'
# Tên file model sẽ được tải về
DOWNLOADED_MODEL_NAME = 'best_weights_model.h5' 

# !!! ĐẢM BẢO THAY THẾ ID FILE DƯỚI ĐÂY BẰNG ID CỦA BẠN !!!
# ID file best_weights_model.h5 trên Google Drive của bạn (phải chia sẻ công khai)
DRIVE_FILE_ID = '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw' 

MODEL_PATH = os.path.join(MODEL_FOLDER, DOWNLOADED_MODEL_NAME) 

os.makedirs(MODEL_FOLDER, exist_ok=True)

# =============================
# Tải model từ Google Drive
# =============================
def download_model_from_drive():
    # Kiểm tra cấu hình bắt buộc
    if not gdown or DRIVE_FILE_ID == '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw':
        print("⚠️ Gdown chưa được import hoặc DRIVE_FILE_ID chưa được cập nhật trong app.py.")
        return None

    # Nếu file đã tồn tại, không tải lại
    if os.path.exists(MODEL_PATH):
        print("✅ File model đã tồn tại. Bỏ qua tải xuống.")
        return MODEL_PATH

    print(f"🔧 Bắt đầu tải file model từ Google Drive (ID: {DRIVE_FILE_ID})...")
    
    try:
        # Tải xuống file từ Drive
        gdown.download(
            id=DRIVE_FILE_ID, 
            output=MODEL_PATH, 
            quiet=False, 
            fuzzy=True,
            use_cookies=False
        )
        if os.path.exists(MODEL_PATH):
            print(f"✅ Tải model thành công. Kích thước: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
            return MODEL_PATH
        else:
            print("❌ Lỗi: gdown không tạo ra file model. Kiểm tra quyền chia sẻ Drive (phải là 'Anyone with the link').")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi tải model từ Drive: {e}")
        return None

# =============================
# Load model khi khởi động
# =============================
model = None
if load_model: 
    MODEL_FILE_PATH = download_model_from_drive()
    if MODEL_FILE_PATH:
        try:
            # Load model từ file .h5 đã tải về
            model = load_model(MODEL_FILE_PATH) 
            print("✅ Model y tế đã load thành công.")
        except Exception as e:
            print("❌ Lỗi khi load model sau khi tải:", e)
            print("❌ Lỗi này có thể do file model bị hỏng hoặc lỗi TF/Keras. Vui lòng kiểm tra lại file gốc.")
            model = None
else:
    print("❌ Model không thể load do Keras/Tensorflow không được import.")
    

# =============================
# Các route Flask (Không thay đổi)
# =============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập Dashboard.', 'warning')
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/emr_profile')
def emr_profile():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    return render_template('emr_profile.html')

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

@app.route('/emr_prediction')
def emr_prediction():
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    return render_template('emr_prediction.html')

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
        # Thông báo lỗi load model cho người dùng
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

@app.route('/logout')
def logout():
    session.clear()
    flash('Bạn đã đăng xuất.', 'info')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
