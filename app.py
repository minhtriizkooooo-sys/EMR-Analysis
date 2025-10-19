import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from functools import wraps

# --- CẤU HÌNH ỨNG DỤNG ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_DATA = {'csv', 'xlsx', 'xls'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}

# ID mô hình giả lập
DRIVE_MODEL_FILE_ID = os.getenv('DRIVE_MODEL_FILE_ID', '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw')
LOCAL_MODEL_CACHE = 'best_weights_model.h5'

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_for_flash_and_session')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DANH SÁCH ẢNH CỐ ĐỊNH ---
NODULE_IMAGES = [
    'Đõ Kỳ Sỹ_1.3.10001.1.1.jpg',
    'Lê Thị Hải_1.3.10001.1.1.jpg',
    'Nguyễn Khoa Luân_1.3.10001.1.1.jpg',
    'Nguyễn Thanh Xuân_1.3.10002.2.2.jpg',
    'Phạm Chí Thanh_1.3.10002.2.2.jpg',
    'Trần Khôi_1.3.10001.1.1.jpg'
]

NON_NODULE_IMAGES = [
    'Nguyễn Danh Hạnh_1.3.10001.1.1.jpg',
    'Nguyễn Thị Quyến_1.3.10001.1.1.jpg',
    'Thái Kim Thư_1.3.10002.2.2.jpg',
    'Võ Thị Ngọc_1.3.10001.1.1.jpg'
]

# --- MODEL GIẢ LẬP ---
EMR_MODEL = None
MODEL_LOAD_SUCCESS = False
DEPENDENCY_ERROR = None

def download_blob(file_id, destination_file_name):
    try:
        with open(destination_file_name, 'w') as f:
            f.write("MOCK_MODEL_DATA")
        return True
    except Exception as e:
        raise Exception(f"Lỗi ghi file mock: {e}")

def check_dependencies_and_load_model():
    global EMR_MODEL, MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR
    try:
        download_blob(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE)
        EMR_MODEL = {"status": "Loaded", "id": DRIVE_MODEL_FILE_ID}
        MODEL_LOAD_SUCCESS = True
    except Exception as e:
        DEPENDENCY_ERROR = f"Lỗi tải model: {e}"
        MODEL_LOAD_SUCCESS = False

def get_model_status():
    if not MODEL_LOAD_SUCCESS:
        check_dependencies_and_load_model()
    return MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR

# --- HỖ TRỢ ---
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('logged_in') is not True:
            flash('Vui lòng đăng nhập.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

def mock_predict_with_model(image_filepath):
    classification_choice = np.random.choice(['Ung thư (Nodule)', 'Không ung thư (Non-nodule)'])
    confidence = np.random.uniform(0.75, 0.88)
    recommendation = "Tiếp tục theo dõi định kỳ." if "Không" in classification_choice else "Cần sinh thiết khẩn cấp."
    return classification_choice, confidence, recommendation

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('userID')
    password = request.form.get('password')
    if user_id and password:
        session['logged_in'] = True
        session['user_id'] = user_id
        return redirect(url_for('dashboard'))
    flash('Đăng nhập thất bại.', 'danger')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/emr_profile')
@login_required
def emr_profile():
    return render_template('emr_profile.html', summary=None, filename=None)

@app.route('/upload_emr', methods=['POST'])
@login_required
def upload_emr():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('Vui lòng chọn file hợp lệ.', 'warning')
        return redirect(url_for('emr_profile'))

    file = request.files['file']
    if allowed_file(file.filename, ALLOWED_EXTENSIONS_DATA):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            summary_html = f"<p>Đã xử lý {len(df)} dòng.</p>"
            os.remove(filepath)
            return render_template('emr_profile.html', summary=summary_html, filename=filename)
        except Exception as e:
            flash(f'Lỗi xử lý file: {e}', 'danger')
            return redirect(url_for('emr_profile'))
    else:
        flash('Chỉ chấp nhận file CSV/XLSX/XLS.', 'danger')
        return redirect(url_for('emr_profile'))

@app.route('/emr_prediction')
@login_required
def emr_prediction():
    is_loaded, error = get_model_status()
    if not is_loaded:
        flash(error or 'Lỗi model AI.', 'danger')
    return render_template('emr_prediction.html', result=None, image_name=None)

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    is_loaded, error = get_model_status()
    if not is_loaded:
        flash("Model chưa sẵn sàng: " + (error or ""), 'danger')
        return redirect(url_for('emr_prediction'))

    if 'image' not in request.files or request.files['image'].filename == '':
        flash('Chưa chọn ảnh.', 'warning')
        return redirect(url_for('emr_prediction'))

    file = request.files['image']
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)

            # --- XỬ LÝ ẢNH CỐ ĐỊNH ---
            if filename in NODULE_IMAGES:
                classification_choice = 'Ung thư (Nodule)'
                confidence = np.random.uniform(0.97, 0.99)
                recommendation = "CẦN SINH THIẾT KHẨN CẤP: Kết quả sơ bộ gợi ý khối u ác tính. Cần hội chẩn chuyên sâu."
                source = "Dự đoán cố định (Nodule)"
            elif filename in NON_NODULE_IMAGES:
                classification_choice = 'Không ung thư (Non-nodule)'
                confidence = np.random.uniform(0.95, 0.98)
                recommendation = "THEO DÕI ĐỊNH KỲ: Không phát hiện khối u ác tính."
                source = "Dự đoán cố định (Non-nodule)"
            else:
                classification_choice, confidence, recommendation = mock_predict_with_model(filepath)
                source = "Dự đoán bằng model mô phỏng"

            result_items = {
                'Phân loại (AI)': classification_choice,
                'Độ tin cậy': f"{confidence * 100:.2f}%",
                'Khuyến nghị': recommendation,
                'Nguồn dự đoán': source
            }

            result_html = "<ul class='list-none space-y-3 p-4 bg-gray-50 rounded-lg'>"
            for key, value in result_items.items():
                icon = '<i class="fas fa-check-circle text-green-500 mr-2"></i>' if 'Non-nodule' in value else (
                        '<i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>' if 'Nodule' in value else
                        '<i class="fas fa-info-circle text-blue-500 mr-2"></i>')
                result_html += f"<li class='flex items-start'><div>{icon}</div><div class='flex-1'><strong>{key}:</strong> {value}</div></li>"
            result_html += "</ul>"

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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- CHẠY APP ---
if __name__ == '__main__':
    app.run(debug=True)
