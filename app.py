import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from functools import wraps

# --- CẤU HÌNH ỨNG DỤNG VÀ THƯ MỤC ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_DATA = {'csv', 'xlsx', 'xls'}
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}

# Thiết lập ID mô hình Drive (Mocked)
DRIVE_MODEL_FILE_ID = os.getenv('DRIVE_MODEL_FILE_ID', '1EAZibH-KDkTB09IkHFCvE-db64xtfJZw')
LOCAL_MODEL_CACHE = 'best_weights_model.h5'

app = Flask(__name__)
# Thiết lập session secret key
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_for_flash_and_session') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DANH SÁCH ẢNH CỐ ĐỊNH KẾT QUẢ DỰ ĐOÁN ---
# Ảnh phải luôn cho kết quả Nodule (Ung thư)
NODULE_IMAGES = [
    'Đõ Kỳ Sỹ_1.3.10001.1.1.jpg', 
    'Lê Thị Hải_1.3.10001.1.1.jpg', 
    'Nguyễn Khoa Luân_1.3.10001.1.1.jpg', 
    'Nguyễn Thanh Xuân_1.3.10002.2.2.jpg', 
    'Phạm Chí Thanh_1.3.10002.2.2.jpg', 
    'Trần Khôi_1.3.10001.1.1.jpg'
]
# Ảnh phải luôn cho kết quả Non-nodule (Không ung thư)
NON_NODULE_IMAGES = [
    'Nguyễn Danh Hạnh_1.3.10001.1.1.jpg', 
    'Nguyễn Thị Quyến_1.3.10001.1.1.jpg', 
    'Thái Kim Thư_1.3.10002.2.2.jpg', 
    'Võ Thị Ngọc_1.3.10001.1.1.jpg'
]

# --- 1. XỬ LÝ LOAD MODEL (LAZY LOADING MOCKUP) ---

EMR_MODEL = None
MODEL_LOAD_SUCCESS = False
DEPENDENCY_ERROR = None

def download_blob(file_id, destination_file_name):
    """Giả lập hàm tải file từ Google Drive."""
    # Giả lập tạo file model rỗng thành công tại local
    try:
        with open(destination_file_name, 'w') as f:
            f.write("MOCK_MODEL_DATA")
        print(f"INFO: Successfully mocked download to local cache: {destination_file_name}")
        return True
    except Exception as e:
        raise Exception(f"Lỗi khi ghi file mock: {e}")

def check_dependencies_and_load_model():
    """Kiểm tra dependency và tải mô hình AI từ Google Drive ID (Mocked)."""
    global EMR_MODEL, MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR

    try:
        download_blob(DRIVE_MODEL_FILE_ID, LOCAL_MODEL_CACHE)
        EMR_MODEL = {"status": "Loaded from Drive ID (Mocked)", "id": DRIVE_MODEL_FILE_ID}
        MODEL_LOAD_SUCCESS = True
        print(f"AI Model loaded successfully (Mocked). ID: {DRIVE_MODEL_FILE_ID}")
    except Exception as e:
        DEPENDENCY_ERROR = f"LỖI TẢI/LOAD MODEL: {e}."
        print(f"FATAL ERROR: {DEPENDENCY_ERROR}")
        MODEL_LOAD_SUCCESS = False

def get_model_status():
    global EMR_MODEL, MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR
    if not MODEL_LOAD_SUCCESS and EMR_MODEL is None:
        check_dependencies_and_load_model()
    return MODEL_LOAD_SUCCESS, DEPENDENCY_ERROR

# --- 2. HÀM HỖ TRỢ CHUNG ---

def allowed_file(filename, allowed_extensions):
    """Kiểm tra đuôi file hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def login_required(f):
    """Decorator kiểm tra đăng nhập."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('logged_in') is not True:
            flash('Vui lòng đăng nhập để truy cập trang này.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def generate_html_table(df):
    """Tạo bảng HTML tóm tắt và mô phỏng báo cáo chuyên sâu từ DataFrame với màu chữ rõ ràng."""
    # 1. Bảng 5 dòng đầu tiên (Đã sửa màu chữ rõ ràng hơn)
    head_html = df.head(5).to_html(
        classes='w-full text-sm text-left text-gray-900 shadow-md sm:rounded-lg', 
        border=0,
        index=False
    )

    # 2. Bảng Tóm tắt thống kê (Đã sửa màu chữ rõ ràng hơn)
    desc_df = df.describe(include='all').T.reset_index()
    desc_df.columns = ['Thuộc tính', 'Đếm', 'Giá trị duy nhất', 'Giá trị phổ biến', 'Tần suất', 'Trung bình', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    desc_df = desc_df.fillna('-').iloc[:, :6]
    
    desc_html = '<div class="overflow-x-auto"><table class="w-full text-sm text-left shadow-md sm:rounded-lg">'
    desc_html += '<thead class="text-xs text-white uppercase bg-indigo-700">'
    desc_html += '<tr>' + ''.join([f'<th scope="col" class="px-4 py-3">{col}</th>' for col in desc_df.columns]) + '</tr>'
    desc_html += '</thead><tbody>'
    
    for index, row in desc_df.iterrows():
        bg_class = 'bg-white border-b hover:bg-gray-50' if index % 2 == 0 else 'bg-gray-50 border-b hover:bg-gray-100'
        desc_html += f'<tr class="{bg_class}">'
        for col in desc_df.columns:
            # Đảm bảo màu chữ là text-gray-900
            if col == 'Thuộc tính':
                desc_html += f'<th scope="row" class="px-4 py-4 font-medium text-gray-900 whitespace-nowrap">{row[col]}</th>'
            else:
                desc_html += f'<td class="px-4 py-4 text-gray-900">{row[col]}</td>' 
        desc_html += '</tr>'
        
    desc_html += '</tbody></table></div>'

    # 3. MÔ PHỎNG BÁO CÁO CHUYÊN SÂU
    total_rows = len(df)
    missing_rate = np.random.uniform(5, 15)
    outlier_cols = np.random.choice(df.columns, size=np.random.randint(1, 4), replace=False)
    
    advanced_report = f"""
    <div class="mt-8 p-4 bg-blue-50 border-l-4 border-blue-500 text-blue-800 rounded-lg">
        <h3 class="text-xl font-semibold text-blue-700 mb-3"><i class="fas fa-microscope mr-2"></i> Phân tích Chuyên sâu (AI Mockup)</h3>
        <ul class="list-disc ml-5 space-y-2">
            <li><strong>Tổng số bệnh án:</strong> {total_rows}</li>
            <li><strong>Tỷ lệ thiếu dữ liệu (TB):</strong> {missing_rate:.2f}% (Cần làm sạch dữ liệu trước khi huấn luyện mô hình sâu).</li>
            <li><strong>Phát hiện Giá trị Bất thường:</strong> Phát hiện {len(outlier_cols)} cột có giá trị ngoại lai đáng kể ({', '.join(outlier_cols)}).</li>
            <li><strong>Khuyến nghị:</strong> Cần xử lý outliers và chuẩn hóa dữ liệu số (Numerical Standardization) trước khi đưa vào mô hình dự đoán.</li>
        </ul>
    </div>
    """
    # Sửa màu chữ cho tiêu đề
    html = f"""
        <h3 class="text-xl font-semibold text-gray-900 mb-4">5 dòng dữ liệu đầu tiên:</h3>
        <div class="overflow-x-auto mb-8">{head_html}</div>
        <h3 class="text-xl font-semibold text-gray-900 mb-4">Tóm tắt thống kê:</h3>
        {desc_html}
        {advanced_report} 
    """
    return html

def mock_predict_with_model(image_filepath):
    """HÀM MÔ PHỎNG: Dự đoán trên ảnh bằng EMR_MODEL đã tải (mô phỏng logic của model H5)."""
    
    # Đây là nơi logic tiền xử lý ảnh và gọi model Keras/TensorFlow (model.predict) sẽ xảy ra.
    # Hiện tại, chúng ta mô phỏng kết quả đầu ra của model với độ tin cậy ngẫu nhiên.
    
    # Mô phỏng kết quả đầu ra của model: Ung thư (Nodule) hoặc Không ung thư (Non-nodule)
    classification_choice = np.random.choice([
        'Ung thư (Nodule)', 
        'Không ung thư (Non-nodule)'
    ], 1)[0]
    
    # Mô phỏng độ tin cậy của model
    confidence = np.random.uniform(0.75, 0.88)
    
    if "Non-nodule" in classification_choice:
        recommendation = "Kết quả sơ bộ không phát hiện khối u ác tính. Tiếp tục theo dõi định kỳ."
    elif "Nodule" in classification_choice:
        recommendation = "Cần sinh thiết khẩn cấp để xác nhận và lập kế hoạch điều trị."
        
    return classification_choice, confidence, recommendation

# --- 3. ROUTE CỦA ỨNG DỤNG ---

@app.route('/')
def index():
    """Trang chủ/Đăng nhập."""
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    """Xử lý đăng nhập (MOCK)."""
    user_id = request.form.get('userID')
    password = request.form.get('password')
    
    if user_id and password:
        session['logged_in'] = True
        session['user_id'] = user_id
        # ĐÃ XÓA DÒNG flash('Đăng nhập thành công! Chuyển hướng đến Dashboard.', 'success') theo yêu cầu.
        return redirect(url_for('dashboard')) 

    flash('Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin.', 'danger')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Xử lý đăng xuất (Xóa session)."""
    session.pop('logged_in', None)
    session.pop('user_id', None)
    flash('Bạn đã đăng xuất thành công.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required 
def dashboard():
    """Trang Dashboard chính."""
    return render_template('dashboard.html')


@app.route('/emr_profile')
@login_required 
def emr_profile():
    """Trang phân tích hồ sơ EMR (CSV/Excel)."""
    return render_template('emr_profile.html', summary=None, filename=None)


@app.route('/upload_emr', methods=['POST'])
@login_required 
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

            flash(f'Phân tích file "{filename}" thành công!', 'success') 
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
@login_required 
def emr_prediction():
    """Trang dự đoán EMR chuyên sâu (Phân loại ảnh)."""
    is_loaded, error = get_model_status()
    if not is_loaded:
        flash(error, 'danger')
    
    return render_template('emr_prediction.html', result=None, image_name=None)


@app.route('/upload_image', methods=['POST'])
@login_required 
def upload_image():
    """Xử lý tải lên ảnh và dự đoán."""
    is_loaded, error = get_model_status()
    if not is_loaded:
        flash("Tính năng dự đoán không khả dụng vì Model AI chưa được tải thành công. Lỗi: " + error, 'danger')
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
            
            # --- LOGIC DỰ ĐOÁN ĐẢM BẢO KẾT QUẢ CỐ ĐỊNH ---
            
            if filename in NODULE_IMAGES:
                # Ảnh cố định: Ung thư (Nodule) -> LUÔN CỐ ĐỊNH
                classification_choice = 'Ung thư (Nodule)'
                recommendation = "CẦN SINH THIẾT KHẨN CẤP: Kết quả sơ bộ gợi ý khối u ác tính. Cần hội chẩn chuyên sâu."
                confidence = np.random.uniform(0.97, 0.99)
                source = "Dự đoán Cố định (Hardcode - Nodule)"
            elif filename in NON_NODULE_IMAGES:
                # Ảnh cố định: Không ung thư (Non-nodule) -> LUÔN CỐ ĐỊNH
                classification_choice = 'Không ung thư (Non-nodule)'
                recommendation = "THEO DÕI ĐỊNH KỲ: Kết quả sơ bộ không phát hiện khối u ác tính. Tiếp tục theo dõi."
                confidence = np.random.uniform(0.95, 0.98)
                source = "Dự đoán Cố định (Hardcode - Non-nodule)"
            else:
                # Ảnh KHÔNG cố định: Gọi mô phỏng dự đoán bằng Model H5 (Mô phỏng)
                classification_choice, confidence, recommendation = mock_predict_with_model(filepath)
                source = "Dự đoán bằng Model H5 (Mô phỏng)"
                
            # --- TẠO KẾT QUẢ HIỂN THỊ ---

            mock_predictions = {
                'Phân loại (AI)': classification_choice,
                'Độ tin cậy': f"{confidence * 100:.2f}%",
                'Khuyến nghị': recommendation,
                'Nguồn dự đoán': source, 
                'Model sử dụng': EMR_MODEL['id']
            }
            
            result_html = "<ul class='list-none space-y-3 p-4 bg-gray-50 rounded-lg'>"
            for key, value in mock_predictions.items():
                icon = '<i class="fas fa-check-circle text-green-500 mr-2"></i>' if 'Thành công' in value or 'Non-nodule' in value else ('<i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>' if 'Nodule' in value else '<i class="fas fa-info-circle text-blue-500 mr-2"></i>')
                result_html += f"<li class='flex items-start'><div>{icon}</div><div class='flex-1'><strong>{key}:</strong> {value}</div></li>"
            result_html += "</ul>"
            
            flash(f'Dự đoán ảnh "{filename}" thành công!', 'success')
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
