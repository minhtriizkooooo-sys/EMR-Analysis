FROM python:3.11-slim

# Set thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng 5000
EXPOSE 5000

# Command để chạy app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
