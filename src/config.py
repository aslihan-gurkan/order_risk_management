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
    "order_items_level"  : os.path.join(PROCESSED_PATH, "order_items_level.parquet"),
    "order_level"        : os.path.join(PROCESSED_PATH, "order_level.parquet"),
    "order_level_labeled": os.path.join(PROCESSED_PATH, "order_level_labeled.parquet"),
    "featured"           : os.path.join(PROCESSED_PATH, "featured.parquet"),
    "train"              : os.path.join(PROCESSED_PATH, "train.parquet"),
    "test"               : os.path.join(PROCESSED_PATH, "test.parquet"),
}

# ── Model dosyaları ───────────────────────────────────────────────────────────
MODEL_FILES = {
    "preprocessor": os.path.join(MODELS_PATH, "preprocessor.joblib"),
    "best_model"  : os.path.join(MODELS_PATH, "best_model.joblib"),
    "pipeline"    : os.path.join(MODELS_PATH, "pipeline.joblib"),
    "metrics"     : os.path.join(OUTPUTS_PATH, "model_metrics.json"),
}

# ── Klasörleri oluştur ────────────────────────────────────────────────────────
for _path in [DATA_PATH, PROCESSED_PATH, MODELS_PATH, OUTPUTS_PATH, LOGS_PATH]:
    os.makedirs(_path, exist_ok=True) # klasör yoksa oluştur, varsa dokunma 

 # repoyu ilk klonladığında data/processed, models, logs klasörleri olmayacak 
 # çünkü .gitignore'a ekledik, Git'e gitmiyor. Kod çalışınca "klasör bulunamadı" hatası alır.
 # Bu satır sayesinde kim main.py çalıştırırsa çalıştırsın, klasörler otomatik oluşuyor. Elle oluşturmasına gerek kalmıyor.
    

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_USER     = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB       = os.getenv("POSTGRES_DB")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")


def get_postgres_url() -> str:
    """
    PostgreSQL bağlantı URL'ini üretir.

    Neden ?
        Config import edildiğinde hemen DB kontrolü yapmayız.
        DB sadece gerçekten kullanılacağı zaman kontrol edilir.
    """
    if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]):
        raise ValueError(
            "PostgreSQL bilgileri eksik! "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB tanımlı olmalı."
        )

    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


# ── Order status definitions ──────────────────────────────────────────────────
MODELING_ORDER_STATUSES = ["delivered", "canceled", "unavailable"]
PROBLEMATIC_ORDER_STATUSES = ["canceled", "unavailable"]

# ── Label thresholds ────────────────────────────────────────────────
DISSATISFACTION_REVIEW_THRESHOLD = 2    # review_score <= 2
DELIVERY_DELAY_THRESHOLD         = 7    # delay > 7 days
# LOGISTICS_DELAY_THRESHOLD     = 5     # gün
# SELLER_RETURN_RATE_THRESHOLD  = 0.35  # oran


# Reference date — max date in dataset + 1 day
# Don't use datetime.now() with historical datasets
REFERENCE_DATE = None  # will be set dynamically from data

#temporal filter - # EDA sonucunda edge period olarak görülen düşük hacimli / incomplete dönemler çıkarılır.
# Bu filtre model training datasını daha stabil hale getirmek için kullanılır.
TRAIN_START_DATE = os.getenv("TRAIN_START_DATE", "2017-01-01")
TRAIN_END_DATE = os.getenv("TRAIN_END_DATE", "2018-09-01")
APPLY_TEMPORAL_FILTER = os.getenv("APPLY_TEMPORAL_FILTER", "true").lower() == "true"
APPLY_SMOTE = os.getenv("APPLY_SMOTE", "false").lower() == "true"