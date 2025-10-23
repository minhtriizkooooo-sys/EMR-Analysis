FROM python:3.11

WORKDIR /app

# Install system dependencies for tensorflow-cpu and other libraries
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
