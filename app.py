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
    # Chỉ import khi cần thiết
    from tensorflow.keras.models import load_model 
except ImportError:
    pass

# --- CẤU HÌNH ỨNG DỤNG VÀ THƯ MỤC ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_DATA = {'csv', 'xlsx', 'xls'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}

# --- CẤU HÌNH GOOGLE DRIVE FILE ID BẰNG BIẾN MÔI TRƯỜNG --- 
DRIVE_MODEL_FILE_ID = os.getenv('DRIVE_MODEL_FILE_ID', '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw') 
LOCAL_MODEL_CACHE = 'best_weights_model.h5' 

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_for_flash') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. XỬ LÝ LỖI IMPORT VÀ LOAD MODEL TỪ DRIVE ID ---

EMR_MODEL = None
MODEL_LOAD_SUCCESS = False
DEPENDENCY_ERROR = None

def download_blob(file_id, destination_file_name):
    """
    Giả lập hàm tải file từ Google Drive. 
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
    """Kiểm tra dependency và tải mô hình AI từ Google Drive ID (Mocked)."""
    global EMR_MODEL, MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR
    
    # 1. Kiểm tra Dependency AI (TensorFlow/Keras)
    try:
        from tensorflow.keras.models import load_model 
        print("Dependency check successful: TensorFlow/Keras found.")
    except ImportError as e:
        DEPENDENCY_ERROR = f"Lỗi thư viện AI: {e}. Vui lòng cài đặt thư viện TensorFlow/Keras để chạy model thực."
        MODEL_LOAD_SUCCESS = False
        print(f"ERROR: {DEPENDENCY_ERROR}")
        return
        
    # 2. Thực hiện tải file model từ Drive ID (hoặc Mocking)
    try:
        download_blob(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE)
        
        # --- MOCKING: Giả lập tải model thành công ---
        EMR_MODEL = {"status": "Loaded from Drive ID (Mocked)", "id": DRIVE_MODEL_FILE_ID}
        MODEL_LOAD_SUCCESS = True
        print(f"AI Model loaded successfully (Mocked). ID: {DRIVE_MODEL_FILE_ID}")

    except Exception as e:
        DEPENDENCY_ERROR = f"LỖI TẢI/LOAD MODEL TỪ DRIVE ID: {e}. Vui lòng kiểm tra ID file và xác thực Drive API."
        print(f"FATAL ERROR: {DEPENDENCY_ERROR}")
        MODEL_LOAD_SUCCESS = False

# Thực hiện kiểm tra và tải model khi ứng dụng khởi động
check_dependencies_and_load_model()

# --- HÀM HỖ TRỢ CHUNG ---

def allowed_file(filename, allowed_extensions):
    """Kiểm tra đuôi file hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_html_table(df):
    """Tạo bảng HTML tóm tắt từ DataFrame với định dạng Tailwind."""
    # 1. Bảng 5 dòng đầu tiên
    head_html = df.head(5).to_html(
        classes='w-full text-sm text-left text-gray-700 dark:text-gray-200 shadow-md sm:rounded-lg', 
        border=0, 
        index=False
    )
    
    # 2. Bảng Tóm tắt thống kê
    desc_df = df.describe(include='all').T.reset_index()
    desc_df.columns = ['Thuộc tính', 'Đếm', 'Giá trị duy nhất', 'Giá trị phổ biến', 'Tần suất', 'Trung bình', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    desc_df = desc_df.fillna('-').iloc[:, :6]
    
    # Tạo HTML thủ công để dễ kiểm soát styling Tailwind
    desc_html = '<div class="overflow-x-auto"><table class="w-full text-sm text-left text-gray-700 dark:text-gray-200 shadow-md sm:rounded-lg">'
    desc_html += '<thead class="text-xs text-white uppercase bg-indigo-700">'
    desc_html += '<tr>' + ''.join([f'<th scope="col" class="px-4 py-3">{col}</th>' for col in desc_df.columns]) + '</tr>'
    desc_html += '</thead><tbody>'
    
    for index, row in desc_df.iterrows():
        bg_class = 'bg-white border-b hover:bg-gray-50' if index % 2 == 0 else 'bg-gray-50 border-b hover:bg-gray-100'
        desc_html += f'<tr class="{bg_class}">'
        for col in desc_df.columns:
            if col == 'Thuộc tính':
                desc_html += f'<th scope="row" class="px-4 py-4 font-medium text-gray-900 whitespace-nowrap">{row[col]}</th>'
            else:
                desc_html += f'<td class="px-4 py-4">{row[col]}</td>'
        desc_html += '</tr>'
        
    desc_html += '</tbody></table></div>'

    html = f"""
        <h3 class="text-xl font-semibold text-gray-700 mb-4">5 dòng dữ liệu đầu tiên:</h3>
        <div class="overflow-x-auto mb-8">{head_html}</div>
        <h3 class="text-xl font-semibold text-gray-700 mb-4">Tóm tắt thống kê:</h3>
        {desc_html}
    """ 
    return html

# --- ROUTE CỦA ỨNG DỤNG ---

@app.route('/')
def index():
    """Trang chủ/Đăng nhập (index.html). Endpoint là 'index'."""
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    """Xử lý đăng nhập (MOCK) và điều hướng đến Dashboard."""
    # flash('Đăng nhập thành công! Chuyển hướng đến Dashboard.', 'success') # Đã loại bỏ flash thành công
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Xử lý đăng xuất (MOCK)."""
    # flash('Bạn đã đăng xuất thành công.', 'success') # Đã loại bỏ flash thành công
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Trang Dashboard chính (dashboard.html). Endpoint là 'dashboard'."""
    return render_template('dashboard.html')


@app.route('/emr_profile')
def emr_profile():
    """Trang phân tích hồ sơ EMR (CSV/Excel). Endpoint là 'emr_profile'."""
    return render_template('emr_profile.html', summary=None, filename=None)


@app.route('/upload_emr', methods=['POST'])
def upload_emr():
    """Xử lý tải lên file EMR (CSV/Excel) và phân tích."""
    if 'file' not in request.files:
        flash('Không có phần file trong request.', 'danger')
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

            # flash(f'Phân tích file "{filename}" thành công!', 'success') # Đã loại bỏ flash thành công
            return render_template('emr_profile.html', summary=summary_html, filename=filename)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Lỗi khi xử lý file EMR: {e}', 'danger')
            return redirect(url_for('emr_profile'))
    else:
        flash('Định dạng file không được hỗ trợ. Chỉ chấp nhận CSV, XLSX, XLS.', 'danger')
        return redirect(url_for('emr_profile'))


@app.route('/emr_prediction')
def emr_prediction():
    """Trang dự đoán EMR chuyên sâu (Phân loại ảnh)."""
    # Nếu model không load được, flash lỗi ngay khi vào trang
    if not MODEL_LOAD_SUCCESS:
        flash(DEPENDENCY_ERROR, 'danger') 
    
    return render_template('emr_prediction.html', result=None, image_name=None)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Xử lý tải lên ảnh và dự đoán."""
    if not MODEL_LOAD_SUCCESS:
        flash("Tính năng dự đoán không khả dụng vì Model AI chưa được tải thành công.", 'danger')
        return redirect(url_for('emr_prediction'))

    if 'image' not in request.files:
        flash('Không có file ảnh nào được tải lên.', 'danger')
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
                'Model sử dụng': EMR_MODEL['id'] 
            }
            
            result_html = "<ul>"
            for key, value in mock_predictions.items():
                result_html += f"<li><strong>{key}:</strong> {value}</li>"
            result_html += "</ul>"
            
            # flash(f'Dự đoán ảnh "{filename}" thành công!', 'success') # Đã loại bỏ flash thành công
            return render_template('emr_prediction.html', result=result_html, image_name=filename)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
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
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
