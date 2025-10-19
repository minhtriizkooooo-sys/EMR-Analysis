import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

# Cố gắng import các thư viện AI và Google API Client (chỉ để kiểm tra sự tồn tại)
try:
    import googleapiclient.discovery 
except ImportError:
    pass 
try:
    from tensorflow.keras.models import load_model 
except ImportError:
    pass

# --- CẤU HÌNH ỨNG DỤNG VÀ THƯ MỤC ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_DATA = {'csv', 'xlsx', 'xls'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}

# --- CẤU HÌNH GOOGLE DRIVE FILE ID BẰNG BIẾN MÔI TRƯỜNG --- 
DRIVE_MODEL_FILE_ID = os.getenv('DRIVE_MODEL_FILE_ID', '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw') 
LOCAL_MODEL_CACHE = 'best_weights_model.h5' # Tên file cache model trên server

app = Flask(__name__)
# Đảm bảo SECRET_KEY được đặt cho Flash Messages
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_for_flash') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. XỬ LÝ LỖI IMPORT VÀ LOAD MODEL TỪ DRIVE ID ---

# Global variable để chứa model và trạng thái
EMR_MODEL = None
MODEL_LOAD_SUCCESS = False
DEPENDENCY_ERROR = None

def download_blob(file_id, destination_file_name):
    """
    Giả lập hàm tải file từ Google Drive bằng File ID.
    (Trong môi trường thực tế, cần OAuth 2.0 và Google Drive API thật)
    """
    print(f"INFO: Attempting to download model using Drive File ID: {file_id}...")

    # --- MOCKING: Giả lập thành công cho mục đích phát triển local ---
    try:
        # Giả lập tạo file model rỗng thành công tại local
        with open(destination_file_name, 'w') as f:
            f.write("MOCK_MODEL_DATA") 
        print(f"INFO: Successfully mocked download to local cache: {destination_file_name}")
        return True
    except Exception as e:
        raise Exception(f"Lỗi khi ghi file mock: {e}")

def check_dependencies_and_load_model():
    """Kiểm tra dependency và tải mô hình AI từ Google Drive ID."""
    global EMR_MODEL, MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR
    
    # 1. Kiểm tra Dependency AI (TensorFlow/Keras)
    try:
        from tensorflow.keras.models import load_model # Kiểm tra sự tồn tại của thư viện
        print("Dependency check successful: TensorFlow/Keras found.")
    except ImportError as e:
        DEPENDENCY_ERROR = f"Lỗi thư viện AI: {e}. Vui lòng cài đặt thư viện TensorFlow/Keras."
        print(f"ERROR: {DEPENDENCY_ERROR}")
        MODEL_LOAD_SUCCESS = False
        return
        
    # 2. Thực hiện tải file model từ Drive ID (hoặc Mocking)
    try:
        download_blob(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE)
        
        # EMR_MODEL = load_model(LOCAL_MODEL_CACHE) # Dùng lệnh này khi có file model thật
        
        # --- MOCKING: Giả lập tải model thành công ---
        EMR_MODEL = {"status": "Loaded from Drive ID (Mocked)", "id": DRIVE_MODEL_FILE_ID}
        # --- END MOCKING ---

        MODEL_LOAD_SUCCESS = True
        print(f"AI Model loaded successfully using Drive ID: {DRIVE_MODEL_FILE_ID}")

    except Exception as e:
        # Lỗi xảy ra khi tải file từ Drive, hoặc file bị hỏng khi load_model
        DEPENDENCY_ERROR = f"LỖI TẢI/LOAD MODEL TỪ DRIVE ID: {e}. Vui lòng kiểm tra ID file và xác thực Google. Lỗi: {e}"
        print(f"FATAL ERROR: {DEPENDENCY_ERROR}")
        MODEL_LOAD_SUCCESS = False

# Thực hiện kiểm tra và tải model khi ứng dụng khởi động
check_dependencies_and_load_model()

# --- HÀM HỖ TRỢ CHUNG ---

def allowed_file(filename, allowed_extensions):
    """Kiểm tra đuôi file hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_html_table(df):
    """Tạo bảng HTML tóm tắt từ DataFrame."""
    head_html = df.head(5).to_html(classes='data-table', border=0, index=False)
    desc_df = df.describe(include='all').T.reset_index()
    desc_df.columns = ['Thuộc tính', 'Đếm', 'Giá trị duy nhất', 'Giá trị phổ biến', 'Tần suất', 'Trung bình', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    desc_df = desc_df.fillna('-').iloc[:, :6] 
    desc_html = desc_df.to_html(classes='summary-table', border=0, index=False)
    
    html = f"""
        <h3>5 dòng dữ liệu đầu tiên:</h3>
        {head_html}
        <h3>Tóm tắt thống kê:</h3>
        {desc_html}
    """ 
    return html

# --- ROUTE CỦA ỨNG DỤNG ---

@app.route('/')
def dashboard():
    """Trang Dashboard chính. Endpoint là 'dashboard'."""
    # Lỗi BuildError xảy ra vì dashboard.html cố gắng gọi url_for('index')
    # Cần sửa trong file template thành url_for('dashboard') hoặc các endpoint khác
    return render_template('dashboard.html')


@app.route('/emr_profile')
def emr_profile():
    """Trang phân tích hồ sơ EMR (CSV/Excel). Endpoint là 'emr_profile'."""
    return render_template('emr_profile.html', summary=None, filename=None)


@app.route('/upload_emr', methods=['POST'])
def upload_emr():
    """Xử lý tải lên file EMR (CSV/Excel) và phân tích. Endpoint là 'upload_emr'."""
    if 'file' not in request.files:
        flash('Không có phần file trong request.', 'warning')
        return redirect(url_for('emr_profile'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn file.', 'warning')
        return redirect(url_for('emr_profile'))

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_DATA):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            summary_html = generate_html_table(df)
            
            os.remove(filepath)

            flash(f'Phân tích file "{filename}" thành công!', 'success')
            return render_template('emr_profile.html', summary=summary_html, filename=filename)

        except Exception as e:
            flash(f'Lỗi khi xử lý file EMR: {e}', 'danger')
            return redirect(url_for('emr_profile'))
    else:
        flash('Định dạng file không được hỗ trợ. Chỉ chấp nhận CSV, XLSX, XLS.', 'danger')
        return redirect(url_for('emr_profile'))


@app.route('/emr_prediction')
def emr_prediction():
    """Trang dự đoán EMR chuyên sâu (Phân loại ảnh). Endpoint là 'emr_prediction'."""
    
    # Xử lý lỗi model khi vào trang
    if not MODEL_LOAD_SUCCESS:
        flash(DEPENDENCY_ERROR, 'danger') 
    
    return render_template('emr_prediction.html', result=None, image_name=None)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Xử lý tải lên ảnh và dự đoán. Endpoint là 'upload_image'."""
    if not MODEL_LOAD_SUCCESS:
        flash(DEPENDENCY_ERROR, 'danger')
        return redirect(url_for('emr_prediction'))

    if 'image' not in request.files:
        flash('Không có file ảnh nào được tải lên.', 'warning')
        return redirect(url_for('emr_prediction'))

    file = request.files['image']
    if file.filename == '':
        flash('Chưa chọn file ảnh.', 'warning')
        return redirect(url_for('emr_prediction'))

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # --- MOCKING: Giả lập dự đoán ---
            mock_predictions = {
                'Phân loại': np.random.choice(['Ung thư', 'Lành tính', 'Nghi ngờ'], 1)[0],
                'Độ tin cậy': f"{np.random.uniform(0.85, 0.99):.2f}%",
                'Khuyến nghị': "Cần hội chẩn chuyên sâu với bác sĩ chẩn đoán hình ảnh.",
                'Model sử dụng': EMR_MODEL['id'] # Sử dụng ID file
            }
            
            # Tạo HTML cho kết quả
            result_html = "<ul>"
            for key, value in mock_predictions.items():
                result_html += f"<li><strong>{key}:</strong> {value}</li>"
            result_html += "</ul>"
            
            flash(f'Dự đoán ảnh "{filename}" thành công!', 'success')
            return render_template('emr_prediction.html', result=result_html, image_name=filename)

        except Exception as e:
            flash(f'Lỗi khi xử lý hoặc dự đoán ảnh: {e}', 'danger')
            return redirect(url_for('emr_prediction'))
    else:
        flash('Định dạng ảnh không được hỗ trợ.', 'danger')
        return redirect(url_for('emr_prediction'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve file ảnh đã tải lên."""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Đảm bảo thư mục uploads tồn tại
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Chạy ứng dụng
    app.run(debug=True)
