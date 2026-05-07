"""
Proje genelinde kullanılan tüm ayarlar buradan okunur.
Hiçbir dosyaya sabit değer (hardcode) yazılmaz, hepsi buradan gelir.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Ortam ────────────────────────────────────────────────────────────────────
ENV       = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Yollar ───────────────────────────────────────────────────────────────────
DATA_PATH      = os.getenv("DATA_PATH",      "data/raw")
PROCESSED_PATH = os.getenv("PROCESSED_PATH", "data/processed")
MODELS_PATH    = os.getenv("MODELS_PATH",    "models")
OUTPUTS_PATH   = os.getenv("OUTPUTS_PATH",   "data/outputs")
LOGS_PATH      = os.getenv("LOGS_PATH",      "logs")

# ── Model parametreleri ───────────────────────────────────────────────────────
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE    = float(os.getenv("TEST_SIZE",  0.2))
N_TRIALS     = int(os.getenv("N_TRIALS",     50))

# ── Ham veri dosyaları ────────────────────────────────────────────────────────
FILES = {
    "orders"   : os.path.join(DATA_PATH, "olist_orders_dataset.csv"),
    "items"    : os.path.join(DATA_PATH, "olist_order_items_dataset.csv"),
    "reviews"  : os.path.join(DATA_PATH, "olist_order_reviews_dataset.csv"),
    "products" : os.path.join(DATA_PATH, "olist_products_dataset.csv"),
    "customers": os.path.join(DATA_PATH, "olist_customers_dataset.csv"),
    "sellers"  : os.path.join(DATA_PATH, "olist_sellers_dataset.csv"),
    "payments" : os.path.join(DATA_PATH, "olist_order_payments_dataset.csv"),
}

# ── İşlenmiş veri dosyaları ───────────────────────────────────────────────────
PROCESSED_FILES = {
    "merged"  : os.path.join(PROCESSED_PATH, "merged.parquet"),
    "featured": os.path.join(PROCESSED_PATH, "featured.parquet"),
    "train"   : os.path.join(PROCESSED_PATH, "train.parquet"),
    "test"    : os.path.join(PROCESSED_PATH, "test.parquet"),
}

# ── Model dosyaları ───────────────────────────────────────────────────────────
MODEL_FILES = {
    "pipeline"   : os.path.join(MODELS_PATH, "pipeline.pkl"),
    "best_model" : os.path.join(MODELS_PATH, "best_model.pkl"),
    "fraud_model": os.path.join(MODELS_PATH, "fraud_model.pkl"),
}

# ── Klasörleri oluştur ────────────────────────────────────────────────────────
for _path in [DATA_PATH, PROCESSED_PATH, MODELS_PATH, OUTPUTS_PATH, LOGS_PATH]:
    os.makedirs(_path, exist_ok=True) # klasör yoksa oluştur, varsa dokunma 

 # repoyu ilk klonladığında data/processed, models, logs klasörleri olmayacak 
 # çünkü .gitignore'a ekledik, Git'e gitmiyor. Kod çalışınca "klasör bulunamadı" hatası alır.
 # Bu satır sayesinde kim main.py çalıştırırsa çalıştırsın, klasörler otomatik oluşuyor. Elle oluşturmasına gerek kalmıyor.
    

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_USER     = os.getenv("POSTGRES_USER",     "iade_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "iade_password123")
POSTGRES_DB       = os.getenv("POSTGRES_DB",       "iade_db")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST",     "db")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT",     "5432")