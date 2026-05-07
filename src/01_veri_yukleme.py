"""
Veri yükleme ve birleştirme modülü.
Olist CSV dosyalarını okur, birleştirir ve parquet olarak kaydeder.
"""
import pandas as pd
from pathlib import Path
from src.config import FILES, PROCESSED_FILES
from src.logger import get_logger

logger = get_logger(__name__)


def load_csv(key: str) -> pd.DataFrame:
    """
    Config'deki FILES dict'inden CSV okur.

    Args:
        key: FILES dict'indeki anahtar (örn. "orders")

    Returns:
        Yüklenen DataFrame

    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    path = FILES[key]

    if not Path(path).exists():
        logger.error(f"Dosya bulunamadi: {path}")
        raise FileNotFoundError(
            f"{path} bulunamadi. "
            f"Dosyayi data/raw/ klasorune koydugunuzdan emin olun."
        )

    df = pd.read_csv(path)
    logger.info(f"{key} yuklendi → {df.shape[0]:,} satir, {df.shape[1]} sutun")
    return df


def validate_dataframe(df: pd.DataFrame, name: str, required_cols: list) -> None:
    """
    DataFrame'in beklenen kolonları içerip içermediğini kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        name: Log mesajı için tablo adı
        required_cols: Olması gereken kolonlar

    Raises:
        ValueError: Kolon eksikse
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"{name} tablosunda eksik kolonlar: {missing}")
        raise ValueError(f"{name} tablosunda eksik kolonlar: {missing}")
    logger.info(f"{name} validasyonu gecti")


def merge_tables(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tüm tabloları order_id üzerinden birleştirir.

    Returns:
        Birleştirilmiş DataFrame
    """
    logger.info("Tablolar birlestiriliyor...")

    # orders + items
    df = orders.merge(items, on="order_id", how="left")
    logger.info(f"orders + items → {df.shape[0]:,} satir")

    # + customers
    df = df.merge(customers, on="customer_id", how="left")
    logger.info(f"+ customers → {df.shape[0]:,} satir")

    # + products
    df = df.merge(products, on="product_id", how="left")
    logger.info(f"+ products → {df.shape[0]:,} satir")

    # + sellers
    df = df.merge(sellers, on="seller_id", how="left")
    logger.info(f"+ sellers → {df.shape[0]:,} satir")

    # + payments (sipariş başına tek satır için aggregate)
    payments_agg = payments.groupby("order_id").agg(
        odeme_turu=("payment_type", "first"),
        taksit_sayisi=("payment_installments", "max"),
        toplam_odeme=("payment_value", "sum"),
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    logger.info(f"+ payments → {df.shape[0]:,} satir")

    # + reviews (sipariş başına en son review)
    reviews_agg = reviews.sort_values("review_creation_date").groupby("order_id").last().reset_index()
    df = df.merge(
        reviews_agg[["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left"
    )
    logger.info(f"+ reviews → {df.shape[0]:,} satir")

    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tarih kolonlarını datetime formatına çevirir.
    """
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Tarih kolonlari duzenlendi")
    return df


def report_missing(df: pd.DataFrame) -> None:
    """
    Eksik değer raporunu loglar.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        logger.info("Eksik deger bulunamadi")
        return

    logger.warning(f"Eksik deger raporu ({len(missing)} kolon):")
    for col, count in missing.items():
        pct = count / len(df) * 100
        logger.warning(f"  {col}: {count:,} ({pct:.1f}%)")


def run() -> pd.DataFrame:
    """
    Ana fonksiyon — pipeline'dan çağrılır.
    Tüm adımları sırayla çalıştırır ve sonucu kaydeder.

    Returns:
        Birleştirilmiş ve temizlenmiş DataFrame
    """
    logger.info("=" * 40)
    logger.info("ADIM 1: Veri yukleme basladi")
    logger.info("=" * 40)

    # 1. CSV'leri yükle
    orders    = load_csv("orders")
    items     = load_csv("items")
    reviews   = load_csv("reviews")
    products  = load_csv("products")
    customers = load_csv("customers")
    sellers   = load_csv("sellers")
    payments  = load_csv("payments")

    # 2. Validasyon — beklenen kolonlar var mı?
    validate_dataframe(orders, "orders", [
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    validate_dataframe(items, "items", [
        "order_id", "product_id", "seller_id", "price", "freight_value"
    ])

    # 3. Birleştir
    df = merge_tables(orders, items, reviews, products, customers, sellers, payments)

    # 4. Tarihleri düzenle
    df = clean_dates(df)

    # 5. Eksik değer raporu
    report_missing(df)

    # 6. Kaydet
    Path(PROCESSED_FILES["merged"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_FILES["merged"], index=False)
    logger.info(f"Birlestirilmis veri kaydedildi → {PROCESSED_FILES['merged']}")
    logger.info(f"Final boyut: {df.shape[0]:,} satir, {df.shape[1]} sutun")

    return df


if __name__ == "__main__":
    run()