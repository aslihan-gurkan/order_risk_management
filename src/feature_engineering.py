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



def add_product_physical_features(order_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ürün fiziksel özelliklerinden order-level feature üretir.

    Neden?
        Ağır, hacimli veya katalog bilgisi zayıf ürünler
        lojistik problem ve müşteri memnuniyetsizliği riskini artırabilir.

    Örnek:
        - ağır ürün: taşıma/hasar/gecikme riski
        - yüksek hacim: lojistik operasyon zorluğu
        - az fotoğraf/açıklama: müşteri beklentisi uyuşmazlığı
    """
    df_items = item_df.copy()

    df_items["product_volume_cm3"] = (
        df_items["product_length_cm"].fillna(0)
        * df_items["product_height_cm"].fillna(0)
        * df_items["product_width_cm"].fillna(0)
    )

    product_features = (
        df_items
        .groupby("order_id", as_index=False)
        .agg(
            total_product_weight_g=("product_weight_g", "sum"),
            avg_product_weight_g=("product_weight_g", "mean"),
            max_product_weight_g=("product_weight_g", "max"),
            total_product_volume_cm3=("product_volume_cm3", "sum"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
            max_product_volume_cm3=("product_volume_cm3", "max"),
            avg_product_photos_qty=("product_photos_qty", "mean"),
            avg_product_description_length=("product_description_lenght", "mean"),
        )
    )

    df = order_df.merge(product_features, on="order_id", how="left")

    logger.info("Ürün fiziksel ve katalog feature'ları oluşturuldu")
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

    df["is_unknown_category"] = (df["main_product_category"] == "unknown").astype(int)

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


def add_payment_features(df: pd.DataFrame, payments_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ham payments tablosundan türetilen gelişmiş ödeme feature'ları.
    """
    # Karışık ödeme tipi kullanan siparişler — boleto + kredi kartı gibi
    payment_type_count = (
        payments_raw.groupby("order_id")["payment_type"]
        .nunique()
        .reset_index()
        .rename(columns={"payment_type": "payment_type_count"})
    )

    # Voucher kullanımı — indirim kuponu = beklenti uyuşmazlığı riski
    voucher_flag = (
        payments_raw.groupby("order_id")["payment_type"]
        .apply(lambda x: int("voucher" in x.values))
        .reset_index()
        .rename(columns={"payment_type": "has_voucher"})
    )

    # Boleto kullanımı — Brezilya'ya özgü, gecikmeli ödeme = iptal riski
    boleto_flag = (
        payments_raw.groupby("order_id")["payment_type"]
        .apply(lambda x: int("boleto" in x.values))
        .reset_index()
        .rename(columns={"payment_type": "has_boleto"})
    )

    df = df.merge(payment_type_count, on="order_id", how="left")
    df = df.merge(voucher_flag, on="order_id", how="left")
    df = df.merge(boleto_flag, on="order_id", how="left")

    df["payment_type_count"] = df["payment_type_count"].fillna(1)
    df["has_voucher"] = df["has_voucher"].fillna(0).astype(int)
    # Brezilya'da boleto ödemesi banka üzerinden yapılır, gecikmesi veya iptali çok daha sık oluşur.
    df["has_boleto"] = df["has_boleto"].fillna(0).astype(int)

    logger.info("Gelişmiş ödeme feature'ları oluşturuldu")
    return df

def add_shipping_urgency_features(df: pd.DataFrame, items_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Kargo son tarihi ile sipariş tarihi arasındaki süre.
    Satıcının kargoya verme için ne kadar vakti var?
    """
    shipping_urgency = (
        items_raw.groupby("order_id")["shipping_limit_date"]
        .min()  # en erken kargo limiti
        .reset_index()
        .rename(columns={"shipping_limit_date": "earliest_shipping_limit"})
    )

    df = df.merge(shipping_urgency, on="order_id", how="left")

    df["shipping_limit_days"] = (
        pd.to_datetime(df["earliest_shipping_limit"]) -
        pd.to_datetime(df["order_purchase_timestamp"])
    ).dt.days

    df["is_urgent_shipping"] = (df["shipping_limit_days"] <= 3).astype(int)
    df = df.drop(columns=["earliest_shipping_limit"])

    logger.info("Kargo aciliyet feature'ları oluşturuldu")
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model sinyal gücünü artıracak ek feature'lar üretir.
    """
    df = df.sort_values("order_purchase_timestamp").copy()

    # 1. Satıcı başına sipariş yoğunluğu — çok siparişli satıcılar daha riskli mi?
    seller_order_velocity = (
        df.groupby("main_seller_id")["order_id"]
        .transform("count")
    )
    df["seller_total_order_count"] = seller_order_velocity

    # 2. Müşteri ilk siparişi mi? — yeni müşteriler farklı risk profiline sahip
    df["is_first_order"] = (df["customer_historical_orders"] == 0).astype(int)

    # 3. Yüksek riskli satıcı flag — seller_historical_problem_rate üst %25
    threshold = df["seller_historical_problem_rate"].quantile(0.75)
    df["is_high_risk_seller"] = (
        df["seller_historical_problem_rate"] >= threshold
    ).astype(int)

    # 4. Yüksek riskli kategori flag
    threshold_cat = df["category_historical_problem_rate"].quantile(0.75)
    df["is_high_risk_category"] = (
        df["category_historical_problem_rate"] >= threshold_cat
    ).astype(int)

    # 5. Satıcı deneyimsizlik flag — az siparişli satıcı
    df["is_new_seller"] = (df["seller_historical_order_count"] <= 5).astype(int)

    # 6. Kargo/fiyat oranı yüksek + çok ürün kombinasyonu — operasyonel yük
    df["heavy_multiitem_order"] = (
        (df["is_multi_item_order"] == 1) &
        (df["freight_ratio"] > 0.3)
    ).astype(int)

    # 7. Uzun tahmini teslimat + yüksek riskli satıcı kombinasyonu
    df["risky_long_delivery"] = (
        (df["estimated_delivery_days"] > 20) &
        (df["is_high_risk_seller"] == 1)
    ).astype(int)

    logger.info("Gelişmiş feature'lar oluşturuldu")
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
    "risk_score",
    "review_creation_date",
    "review_comment_message",
    "review_comment_title",
    "review_missing",

    "delivery_delay_days",
    "is_undelivered",

    "label_low_review",
    "label_very_low_review",
    "label_delivery_delay",
    "label_severe_delivery_delay", # TODO ekle ve analiz et
    "label_problematic_status",

    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_status",
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
    payments_raw = pd.read_parquet(PROCESSED_FILES["payments_raw"])
    items_raw = pd.read_parquet(PROCESSED_FILES["items_raw"])
    items_raw["shipping_limit_date"] = pd.to_datetime(
        items_raw["shipping_limit_date"], errors="coerce"
    )

    logger.info(f"Order-level labeled veri yüklendi -> {order_df.shape}")
    logger.info(f"Item-level veri yüklendi -> {item_df.shape}")

    df = order_df.copy()

    df = add_time_features(df)
    df = add_order_financial_features(df)
    df = add_logistics_features(df)
    df = add_location_features(df, item_df)
    df = add_product_category_features(df, item_df)
    df = add_product_physical_features(df, item_df)
    df = add_historical_risk_features(df)
    df = add_advanced_features(df)
    df = add_payment_features(df, payments_raw)  
    df = add_shipping_urgency_features(df, items_raw) 
    df = drop_leakage_columns(df)

    output_path = PROCESSED_FILES["featured"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("=" * 60)
    logger.info("ADIM 3 tamamlandı")
    logger.info(f"Featured veri kaydedildi -> {output_path}")
    logger.info(f"Final boyut: {df.shape[0]:,} satır, {df.shape[1]} kolon")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run()