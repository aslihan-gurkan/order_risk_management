"""
PostgreSQL bağlantısı ve tablo tanımları.
SQLAlchemy ORM kullanılır.
Tüm DB işlemleri buradan yapılır.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, DateTime, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import get_postgres_url
from src.logger import get_logger

logger = get_logger(__name__)

# ── Bağlantı URL'i ────────────────────────────────────────────────────────────
DATABASE_URL = get_postgres_url()

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Tablo tanımları ───────────────────────────────────────────────────────────
class Prediction(Base):
    """
    Her API tahmini bu tabloya yazılır.
    Modelin gerçek dünya performansını izlemek için kullanılır.
    """
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    order_id            = Column(String, index=True)

    # Girdi özellikleri
    input_features      = Column(JSON)          # gelen sipariş bilgileri

    # Tahmin sonuçları
    iade_olasiligi      = Column(Float)          # 0.0 - 1.0
    iade_tipi           = Column(Integer)        # 0,1,2,3,4
    iade_tipi_aciklama  = Column(String)         # "Meşru iade" gibi
    fraud_risk_skoru    = Column(Float)          # 0 - 100
    fraud_risk_seviyesi = Column(String)         # Düşük/Orta/Yüksek/Kritik

    # Önerilen aksiyon
    aksiyon             = Column(String)

    # Meta
    model_versiyonu     = Column(String, default="1.0.0")
    tahmin_zamani       = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """
    Her model eğitiminin metriklerini saklar.
    Zaman içinde model performansını karşılaştırmak için.
    """
    __tablename__ = "model_metrics"

    id           = Column(Integer, primary_key=True, index=True)
    model_adi    = Column(String)
    f1_score     = Column(Float)
    auc_roc      = Column(Float)
    precision    = Column(Float)
    recall       = Column(Float)
    egitim_tarihi= Column(DateTime, default=datetime.utcnow)
    notlar       = Column(String, nullable=True)


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def get_db():
    """
    FastAPI dependency injection için DB session döner.
    Her istek kendi session'ını açar ve kapatır.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Tablolar yoksa oluşturur."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Veritabani tablolari olusturuldu")
    except Exception as e:
        logger.error(f"Tablo olusturma hatasi: {e}")
        raise


def check_connection() -> bool:
    """DB bağlantısını test eder."""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("PostgreSQL baglantisi basarili")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL baglanti hatasi: {e}")
        return False