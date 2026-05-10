#1
"""
Veri yükleme ve birleştirme modülü.
Tüm Olist CSV dosyalarını okur, validate eder, birleştirir
ve iki farklı grain'de parquet olarak kaydeder.

İki çıktı:
    1. order_items_level.parquet → ürün/satıcı kırılımı için
    2. order_level.parquet       → modelleme için ana tablo

Notlar:
    - orders + items birleşimi item-level veri üretir.
    - Sipariş sayısı için nunique(order_id) kullan, count değil.
    - Referans tarih veri setindeki max tarih + 1 gün olarak atanır.
    - datetime.now() kullanılmaz — historical dataset.
"""
import pandas as pd
from pathlib import Path

import src.config as config
from src.config import FILES, PROCESSED_FILES
from src.logger import get_logger

logger = get_logger(__name__)


# ── Validasyon şemaları ───────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "orders": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": [
        "order_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    ],
    "reviews": [
        "order_id",
        "review_score",
        "review_creation_date",
    ],
    "products": [
        "product_id",
        "product_category_name",
    ],
    "customers": [
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ],
    "sellers": [
        "seller_id",
        "seller_city",
        "seller_state",
    ],
    "payments": [
        "order_id",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
}

# ── Tablo bazında tarih kolonları ─────────────────────────────────────────────
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "items": [
    "shipping_limit_date",
    ],
}


# ── Yükleme ───────────────────────────────────────────────────────────────────
def load_csv(key: str) -> pd.DataFrame:
    """
    Config'deki FILES dict'inden CSV dosyasını okur.

    Args:
        key: FILES dict'indeki anahtar (örn. "orders")

    Returns:
        Yüklenen DataFrame

    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    path = Path(FILES[key])

    if not path.exists():
        logger.error(f"Dosya bulunamadı: {path}")
        raise FileNotFoundError(
            f"{path} bulunamadı. "
            f"Dosyayı data/raw/ klasörüne koyduğunuzdan emin olun."
        )

    df = pd.read_csv(path)
    logger.info(f"{key} yüklendi → {df.shape[0]:,} satır, {df.shape[1]} sütun")
    return df


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Config'deki tüm CSV dosyalarını yükler."""
    return {key: load_csv(key) for key in FILES}


# ── Validasyon ────────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    DataFrame'in beklenen kolonları içerip içermediğini kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        name: Tablo adı — REQUIRED_COLUMNS'daki key ile eşleşmeli

    Raises:
        ValueError: Eksik kolon varsa
    """
    required_cols = REQUIRED_COLUMNS.get(name, [])
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        logger.error(f"{name} tablosunda eksik kolonlar: {missing_cols}")
        raise ValueError(f"{name} tablosunda eksik kolonlar: {missing_cols}")

    logger.info(f"{name} validasyonu geçti ✓")


def validate_all(tables: dict[str, pd.DataFrame]) -> None:
    """Tüm tabloları tek seferde validate eder."""
    logger.info("Tüm tablolar validate ediliyor...")
    for name, df in tables.items():
        validate_dataframe(df, name)


# ── Tarih dönüşümü ────────────────────────────────────────────────────────────
def convert_dates(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Tüm tablolardaki tarih kolonlarını datetime formatına çevirir.

    Tüm dönüşümler burada yapılır — downstream adımlarda tekrar yapılmaz.
    Parquet datetime tiplerini korur, yeniden dönüştürmeye gerek kalmaz.
    """
    for table_name, date_cols in DATE_COLUMNS.items():
        if table_name not in tables:
            continue
        for col in date_cols:
            if col in tables[table_name].columns:
                tables[table_name][col] = pd.to_datetime(
                    tables[table_name][col],
                    errors="coerce",
                )

    logger.info("Tarih kolonları datetime formatına çevrildi")
    return tables


# ── Referans tarih ────────────────────────────────────────────────────────────
def set_reference_date(orders: pd.DataFrame) -> None:
    """
    Global REFERENCE_DATE'i veri setindeki max tarih + 1 gün olarak atar.

    Neden datetime.now() değil?
        Veri seti historical (2016-2018). now() kullanılırsa
        8+ yıllık anlamsız tarih farkları oluşur.
        Tüm modüller config.REFERENCE_DATE'den okur.

    Raises:
        ValueError: Geçerli timestamp bulunamazsa
    """
    max_date = orders["order_purchase_timestamp"].max()

    if pd.isna(max_date):
        raise ValueError(
            "order_purchase_timestamp kolonunda geçerli tarih bulunamadı."
        )

    config.REFERENCE_DATE = max_date + pd.Timedelta(days=1)
    logger.info(f"Referans tarih belirlendi → {config.REFERENCE_DATE.date()}")


# ── Aggregation'lar ───────────────────────────────────────────────────────────
def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Payments tablosunu order bazına indirir.

    payment_type için unique join kullanılır — bilgi kaybı önlenir.
    'first' kullansaydık birden fazla ödeme tipi varsa kaybolurdu.
    payment_count çok ödemeli siparişleri yakalar.
    """
    payments_agg = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            payment_type=(
                "payment_type",
                lambda x: ",".join(sorted(x.dropna().astype(str).unique()))
            ),
            payment_installments_max=("payment_installments", "max"),
            payment_count=("payment_type", "count"),
            total_payment_value=("payment_value", "sum"),
        )
    )

    logger.info(
        f"payments order-level aggregate edildi → {payments_agg.shape[0]:,} satır"
    )
    return payments_agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Reviews tablosunu order bazına indirir.

    Sıralamadan önce review_creation_date datetime olmalı.
    String tarih sıralaması yanlış kronolojik sıra verir.
    Her sipariş için en son yorum alınır.
    review_comment_message opsiyonel — her siparişte olmayabilir.
    """
    selected_cols = ["order_id", "review_score", "review_creation_date"]

    if "review_comment_message" in reviews.columns:
        selected_cols.append("review_comment_message")

    reviews_agg = (
        reviews
        .sort_values("review_creation_date")
        .groupby("order_id", as_index=False)
        .last()[selected_cols]
    )

    logger.info(
        f"reviews order-level aggregate edildi → {reviews_agg.shape[0]:,} satır"
    )
    return reviews_agg


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Items tablosunu order bazına indirir.

    Ürün sayısı, unique satıcı sayısı, toplam fiyat ve kargo
    order-level modelleme için korunur.
    """
    return (
        items
        .groupby("order_id", as_index=False)
        .agg(
            item_count=("product_id", "count"),
            product_count=("product_id", "nunique"),
            seller_count=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
        )
    )


# ── Birleştirme ───────────────────────────────────────────────────────────────
def create_item_level_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Feature engineering için item-level tablo oluşturur.

    Grain: her order_item için bir satır.
    Adım 03'te satıcı ve ürün feature aggregation'ları için kullanılır.
    """
    logger.info("Item-level tablo oluşturuluyor...")

    df = (
        tables["orders"]
        .merge(tables["items"],     on="order_id",    how="left")
        .merge(tables["customers"], on="customer_id", how="left")
        .merge(tables["products"],  on="product_id",  how="left")
        .merge(tables["sellers"],   on="seller_id",   how="left")
    )

    logger.info(
        f"Item-level tablo → {df.shape[0]:,} satır, "
        f"{df['order_id'].nunique():,} unique sipariş"
    )
    return df


def create_order_level_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Modelleme için order-level tablo oluşturur.

    Grain: her sipariş için bir satır.

    Products ve sellers tabloları burada kasıtlı olarak dahil edilmedi.
    Ürün kategorisi ve satıcı feature'ları adım 03'te
    item-level aggregation'lardan hesaplanacak.
    Bu yaklaşım order-level tabloyu temiz tutar ve fan-out join'i önler.
    """
    logger.info("Order-level tablo oluşturuluyor...")

    items_agg    = aggregate_items(tables["items"])
    payments_agg = aggregate_payments(tables["payments"])
    reviews_agg  = aggregate_reviews(tables["reviews"])

    df = (
        tables["orders"]
        .merge(tables["customers"], on="customer_id", how="left")
        .merge(items_agg,           on="order_id",    how="left")
        .merge(payments_agg,        on="order_id",    how="left")
        .merge(reviews_agg,         on="order_id",    how="left")
    )

    logger.info(
        f"Order-level tablo → {df.shape[0]:,} satır, "
        f"{df['order_id'].nunique():,} unique sipariş"
    )
    return df


# ── Eksik değer raporu ────────────────────────────────────────────────────────
def report_missing(df: pd.DataFrame, table_name: str) -> None:
    """Eksik değeri olan tüm kolonları loglar."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        logger.info(f"{table_name}: Eksik değer bulunamadı ✓")
        return

    logger.warning(f"{table_name}: Eksik değer raporu ({len(missing)} kolon)")
    for col, count in missing.items():
        pct = count / len(df) * 100
        logger.warning(f"  {col}: {count:,} ({pct:.1f}%)")


# ── Kaydetme ──────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    DataFrame'i parquet olarak kaydeder.
    Parquet veri tiplerini korur — datetime datetime kalır, int int kalır.
    Downstream adımlarda yeniden dönüştürme gerekmez.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Kaydedildi → {output_path}")


# ── Pipeline giriş noktası ────────────────────────────────────────────────────
def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ana fonksiyon — main.py pipeline'ından çağrılır.

    Returns:
        (order_items_df, order_level_df) tuple'ı
    """
    logger.info("=" * 60)
    logger.info("ADIM 1: Veri yükleme ve temel birleştirme başladı")
    logger.info("=" * 60)

    # Tüm CSV'leri yükle
    tables = load_all_tables()

    # Tüm tabloları validate et
    validate_all(tables)

    # Tarih dönüşümlerini burada yap — downstream'de bir daha yapılmaz
    tables = convert_dates(tables)

    # Referans tarihi veriden ata
    set_reference_date(tables["orders"])

    # İki farklı grain'de tablo oluştur
    order_items_df = create_item_level_table(tables)
    order_level_df = create_order_level_table(tables)

    # Eksik değer raporu
    report_missing(order_items_df, "order_items_level")
    report_missing(order_level_df, "order_level")

    # Kaydet
    save_parquet(order_items_df, PROCESSED_FILES["order_items_level"])
    save_parquet(order_level_df, PROCESSED_FILES["order_level"])

    logger.info("=" * 60)
    logger.info("ADIM 1 tamamlandı ✓")
    logger.info(
        f"Item-level: {order_items_df.shape[0]:,} satır, "
        f"{order_items_df.shape[1]} kolon"
    )
    logger.info(
        f"Order-level: {order_level_df.shape[0]:,} satır, "
        f"{order_level_df.shape[1]} kolon"
    )
    logger.info("=" * 60)

    return order_items_df, order_level_df


if __name__ == "__main__":
    run()