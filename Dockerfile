FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Önce requirements kopyala — layer cache için
# Kod değişirse bile bağımlılıkları tekrar yüklemez
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# FastAPI portu
EXPOSE 8000

# Streamlit portu
EXPOSE 8501

# Varsayılan komut — docker-compose'da override edilecek
CMD ["python", "main.py"]