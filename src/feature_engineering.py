#3
"""
Feature engineering modülü.

Amaç:
    Modelin sipariş tamamlanmadan önce kullanabileceği değişkenleri üretmek.

Kritik prensip:
    Label üretirken kullanılan gelecek bilgileri feature olarak kullanmayız.
    Örneğin:
        - review_score
        - delivery_delay_days
        - order_delivered_customer_date
    model feature'ı olamaz.

Bu modül:
    1. Order-level labeled veriyi okur
    2. Item-level veriden ürün/satıcı/category feature'ları üretir
    3. Zaman bazlı feature'lar çıkarır
    4. Leakage riski olan kolonları ayırır
    5. featured.parquet olarak kaydeder
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_FILES
from src.logger import get_logger

logger = get_logger(__name__)


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def safe_divide(numerator, denominator):
    """
    Sıfıra bölme hatasını önlemek için güvenli bölme yapar.

    Neden gerekli?
        Bazı siparişlerde total_price 0 veya NaN olabilir.
        Direkt bölme yaparsak inf veya hata oluşabilir.
    """
    return numerator / denominator.replace(0, np.nan)


# ── Temel zaman feature'ları ──────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sipariş satın alma tarihinden zaman bazlı feature'lar üretir.

    Bu değişkenler sipariş anında bilinebilir.
    Bu yüzden leakage oluşturmaz.
    """
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    df["is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)
    df["is_night_order"] = df["purchase_hour"].between(0, 6).astype(int)

    logger.info("Zaman feature'ları oluşturuldu")
    return df


# ── Sipariş finansal feature'ları ─────────────────────────────────────────────
def add_order_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sipariş tutarı, kargo oranı ve ödeme davranışı feature'ları üretir.

    Neden önemli?
        Yüksek kargo oranı, çok taksit, çok ürün gibi sinyaller
        müşteri memnuniyetsizliği ve operasyonel riskle ilişkili olabilir.
    """
    df["order_total_value"] = (
        df["total_price"].fillna(0) +
        df["total_freight_value"].fillna(0)
    )

    df["freight_ratio"] = safe_divide(
        df["total_freight_value"].fillna(0),
        df["order_total_value"].replace(0, np.nan)
    ).fillna(0)

    df["avg_item_price"] = safe_divide(
        df["total_price"].fillna(0),
        df["item_count"].replace(0, np.nan)
    ).fillna(0)

    df["is_multi_item_order"] = (df["item_count"].fillna(0) > 1).astype(int)
    df["is_multi_seller_order"] = (df["seller_count"].fillna(0) > 1).astype(int)
    df["is_installment_payment"] = (
        df["payment_installments_max"].fillna(0) > 1
    ).astype(int)

    logger.info("Finansal ve sipariş sepeti feature'ları oluşturuldu")
    return df


# ── Lojistik feature'ları ─────────────────────────────────────────────────────
def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sipariş anında bilinebilecek lojistik feature'lar üretir.

    Önemli:
        Gerçek teslimat tarihi kullanılmaz.
        Sadece satın alma tarihi ve tahmini teslimat tarihi kullanılır.
    """
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] -
        df["order_purchase_timestamp"]
    ).dt.days

    df["approval_delay_hours"] = (
        df["order_approved_at"] -
        df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    df["approval_delay_hours"] = df["approval_delay_hours"].fillna(
        df["approval_delay_hours"].median()
    )

    logger.info("Lojistik feature'ları oluşturuldu")
    return df


# ── Lokasyon feature'ları ─────────────────────────────────────────────────────
def add_location_features(order_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Müşteri ve satıcı lokasyonu üzerinden feature üretir.

    Neden item-level tablo kullanıyoruz?
        Bir siparişte birden fazla satıcı olabilir.
        Seller bilgisi order-level tabloda doğrudan yok.
        Bu yüzden item-level'dan order bazına indiriyoruz.
    """
    seller_location = (
    item_df
    .groupby("order_id", as_index=False)
    .agg(
        main_seller_id=("seller_id", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        seller_state_mode=("seller_state", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        seller_city_nunique=("seller_city", "nunique"),
        seller_state_nunique=("seller_state", "nunique"),
    )
)

    df = order_df.merge(seller_location, on="order_id", how="left")

    df["same_customer_seller_state"] = (
        df["customer_state"] == df["seller_state_mode"]
    ).astype(int)

    logger.info("Lokasyon feature'ları oluşturuldu")
    return df


# ── Ürün kategori feature'ları ────────────────────────────────────────────────
def add_product_category_features(order_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ürün kategorisi bilgisini order-level'a indirir.

    Neden direkt products tablosunu order-level'a joinlemiyoruz?
        Çünkü bir siparişte birden fazla ürün olabilir.
        Direkt join satır çoğaltır.
        Önce item-level'dan order-level aggregation yapıyoruz.
    """
    category_features = (
        item_df
        .groupby("order_id", as_index=False)
        .agg(
            main_product_category=("product_category_name", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
            category_count=("product_category_name", "nunique"),
        )
    )

    df = order_df.merge(category_features, on="order_id", how="left")
    df["main_product_category"] = df["main_product_category"].fillna("unknown")
    df["category_count"] = df["category_count"].fillna(0)

    logger.info("Ürün kategori feature'ları oluşturuldu")
    return df


# ── Historical seller/category risk feature'ları ──────────────────────────────
def add_historical_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Satıcı ve kategori geçmiş problematic oranlarını üretir.

    Leakage önlemi:
        Her hesaplamada shift(1) kullanılır.
        Böylece mevcut siparişin label'ı kendi feature'ına karışmaz.

    Wow effect:
        Bu kısım projeyi basit Miuul ödevinden çıkarıp
        gerçek hayata daha yakın production ML pipeline seviyesine taşır.
    """
    df = df.sort_values("order_purchase_timestamp").copy()

    # seller_id bazlı geçmiş problem oranı
    # Bu satıcının geçmiş siparişlerinde problem oranı yüksek mi?
    # Seller historical risk
    df["seller_historical_problem_count"] = (
        df.groupby("main_seller_id")["problematic_order"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    df["seller_historical_order_count"] = (
        df.groupby("main_seller_id")["order_id"]
        .transform(lambda x: x.shift(1).expanding().count().fillna(0))
    )

    df["seller_historical_problem_rate"] = (
        df["seller_historical_problem_count"] /
        df["seller_historical_order_count"].replace(0, np.nan)
    ).fillna(0)

    # Category historical risk
    df["category_historical_problem_count"] = (
        df.groupby("main_product_category")["problematic_order"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    df["category_historical_order_count"] = (
        df.groupby("main_product_category")["order_id"]
        .transform(lambda x: x.shift(1).expanding().count().fillna(0))
    )

    df["category_historical_problem_rate"] = (
        df["category_historical_problem_count"] /
        df["category_historical_order_count"].replace(0, np.nan)
    ).fillna(0)

    logger.info("Historical risk feature'ları oluşturuldu")
    return df


# ── Leakage kolonlarını temizleme ─────────────────────────────────────────────
def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model eğitiminde kullanılmaması gereken kolonları kaldırır.

    Bu kolonlar label oluşturmak için kullanılabilir ama
    tahmin anında review bilinmez. Modelde kalırlarsa data leakage olur.
    """
    leakage_cols = [
    "review_score",
    "review_creation_date",
    "review_comment_message",
    "review_missing",
    "delivery_delay_days",
    "is_undelivered",
    "label_low_review",
    "label_delivery_delay",
    "label_problematic_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
]

    existing_cols = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=existing_cols)

    logger.info(f"Leakage riski olan kolonlar çıkarıldı: {existing_cols}")
    return df


# ── Ana pipeline ──────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    """
    Feature engineering pipeline'ını çalıştırır.

    Returns:
        Modellemeye hazır feature seti
    """
    logger.info("=" * 60)
    logger.info("ADIM 3: Feature engineering başladı")
    logger.info("=" * 60)

    order_path = PROCESSED_FILES["order_level_labeled"]
    item_path = PROCESSED_FILES["order_items_level"]

    if not Path(order_path).exists():
        raise FileNotFoundError("Önce 02_label_engineering.py çalıştırılmalı.")

    if not Path(item_path).exists():
        raise FileNotFoundError("Önce 01_data_loading.py çalıştırılmalı.")

    order_df = pd.read_parquet(order_path)
    item_df = pd.read_parquet(item_path)

    logger.info(f"Order-level labeled veri yüklendi → {order_df.shape}")
    logger.info(f"Item-level veri yüklendi → {item_df.shape}")

    df = order_df.copy()

    df = add_time_features(df)
    df = add_order_financial_features(df)
    df = add_logistics_features(df)
    df = add_location_features(df, item_df)
    df = add_product_category_features(df, item_df)
    df = add_historical_risk_features(df)
    df = drop_leakage_columns(df)

    output_path = PROCESSED_FILES["featured"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("=" * 60)
    logger.info("ADIM 3 tamamlandı ✓")
    logger.info(f"Featured veri kaydedildi → {output_path}")
    logger.info(f"Final boyut: {df.shape[0]:,} satır, {df.shape[1]} kolon")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run()