FROM python:3.10-slim

WORKDIR /app

# Install sistem dependencies minimal yang dibutuhkan Pillow & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Torch (langsung dari wheel CPU resmi biar cepat dan gak timeout)
RUN pip install --no-cache-dir torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app & model
COPY app.py .
COPY nutrition.json .
COPY models ./models

# Port service (dibaca otomatis oleh Railway/Render)
EXPOSE 8000

# Jalankan FastAPI (tanpa reload, mode produksi)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]