# 2
"""
Label engineering modülü.

Amaç:
    Olist veri setinde gerçek "iade" label'ı olmadığı için
    iş problemiyle uyumlu bir binary hedef değişken üretmek.

Problem:
    Sipariş tamamlanmadan önce sorunlu sipariş riskini tahmin etmek.

Hedef değişken:
    problematic_order
        0 = normal sipariş
        1 = sorunlu sipariş

Sorunlu sipariş tanımı:
    1. review_score <= 2
    VEYA
    2. delivery_delay_days > 7
    VEYA
    3. order_status canceled/unavailable

Önemli:
    Review score ve gerçek teslimat tarihi label üretmek için kullanılır.
    Model feature'ı olarak kullanılmaz.
    Aksi halde data leakage oluşur.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    PROCESSED_FILES,
    MODELING_ORDER_STATUSES,
    PROBLEMATIC_ORDER_STATUSES,
    DISSATISFACTION_REVIEW_THRESHOLD,
    DELIVERY_DELAY_THRESHOLD,
)
from src.logger import get_logger

logger = get_logger(__name__)


def filter_modeling_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece sonucu belli olan siparişleri tutar.

    Neden?
        shipped, invoiced, processing gibi statülerde sipariş sonucu belirsizdir.
        Bu satırlara doğru label veremeyiz.
    """
    before = len(df)

    df = df[df["order_status"].isin(MODELING_ORDER_STATUSES)].copy()

    after = len(df)
    logger.info(
        f"Sonucu belirsiz siparişler çıkarıldı -> "
        f"{before - after:,} satır çıkarıldı, {after:,} satır kaldı"
    )

    logger.info("Kalan order_status dağılımı:")
    for status, count in df["order_status"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {status}: {count:,} ({pct:.1f}%)")

    return df


def calculate_delivery_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Teslimat gecikmesini gün olarak hesaplar.

    delivery_delay_days:
        gerçek teslimat tarihi - tahmini teslimat tarihi

    Pozitif değer:
        Geç teslim edildi.

    Negatif değer:
        Erken teslim edildi.

    Teslim edilmeyen siparişlerde ne yapıyoruz?
        Gerçek teslimat tarihi olmadığı için delivery_delay_days NaN olur.
        Bunları 999 ile dolduruyoruz.

    Neden 999?
        Bu değer label üretiminde "teslim edilmedi" sinyali olarak kullanılır.
        Ancak model feature'ı olarak kullanılmayacak.
        Bu yüzden leakage yaratmaz.

    Ek olarak:
        is_undelivered kolonu oluşturulur.
        Bu kolon da label analizinde kullanılır.
    """
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] -
        df["order_estimated_delivery_date"]
    ).dt.days

    df["is_undelivered"] = df["order_delivered_customer_date"].isna().astype(int)

    undelivered_count = df["is_undelivered"].sum()

    df["delivery_delay_days"] = df["delivery_delay_days"].fillna(999)

    delivered_delays = df.loc[
        df["is_undelivered"] == 0,
        "delivery_delay_days"
    ]

    logger.info(
        f"Teslimat gecikmesi hesaplandı -> "
        f"teslim edilmeyen: {undelivered_count:,}, "
        f"teslim edilen ortalama gecikme: {delivered_delays.mean():.1f} gün"
    )

    return df


def create_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary hedef değişkeni oluşturur.

    Üç sinyal OR mantığıyla birleşir:
        - düşük review score
        - ciddi teslimat gecikmesi
        - canceled/unavailable status

    Neden sinyal kolonları oluşturuyoruz?
        EDA ve sunumda "problematic_order neden oluştu?"
        sorusuna açıklanabilir cevap verebilmek için.
    """
    review_score = df["review_score"].fillna(3)

    df["label_low_review"] = (
        review_score <= DISSATISFACTION_REVIEW_THRESHOLD
    ).astype(int)

    df["label_delivery_delay"] = (
        (df["is_undelivered"] == 0) &
        (df["delivery_delay_days"] > DELIVERY_DELAY_THRESHOLD)
    ).astype(int) 
    # Teslim edildiyse ve geç kaldıysa -> delivery delay
    # Teslim edilmediyse -> status problem

    df["label_problematic_status"] = (
        df["order_status"].isin(PROBLEMATIC_ORDER_STATUSES)
    ).astype(int)

    df["problematic_order"] = (
        (df["label_low_review"] == 1) |
        (df["label_delivery_delay"] == 1) |
        (df["label_problematic_status"] == 1)
    ).astype(int)

    total = len(df)
    problematic = df["problematic_order"].sum()
    normal = total - problematic

    logger.info("Hedef değişken dağılımı:")
    logger.info(f"  Normal      (0): {normal:,} ({normal / total:.1%})")
    logger.info(f"  Problematic (1): {problematic:,} ({problematic / total:.1%})")

    logger.info("Label sinyalleri:")
    logger.info(f"  Düşük review      : {df['label_low_review'].sum():,}")
    logger.info(f"  Teslimat gecikmesi: {df['label_delivery_delay'].sum():,}")
    logger.info(f"  Status problemi   : {df['label_problematic_status'].sum():,}")

    return df


def add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Müşteri bazlı historical risk feature'ları üretir.

    Leakage önlemi:
        shift(1) kullanılır.
        Böylece mevcut sipariş kendi geçmişine dahil edilmez.
    """
    df = df.sort_values("order_purchase_timestamp").copy()

    df["customer_historical_problematic"] = (
        df.groupby("customer_unique_id")["problematic_order"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    df["customer_historical_orders"] = (
        df.groupby("customer_unique_id")["order_id"]
        .transform(lambda x: x.shift(1).expanding().count().fillna(0))
    )

    df["customer_historical_problem_rate"] = (
        df["customer_historical_problematic"] /
        df["customer_historical_orders"].replace(0, np.nan)
    ).fillna(0)

    logger.info("Müşteri historical problem oranı hesaplandı")

    return df


def report_class_balance(df: pd.DataFrame) -> None:
    """
    Hedef değişken dağılımını raporlar.

    Bu bilgi preprocessing aşamasında:
        - class_weight
        - SMOTE
        - threshold tuning
    kararları için kullanılır.
    """
    total = len(df)
    problematic = df["problematic_order"].sum()
    ratio = problematic / total

    if ratio < 0.10:
        logger.warning(
            f"Ciddi class imbalance: problematic oranı {ratio:.1%}"
        )
    elif ratio < 0.30:
        logger.warning(
            f"Hafif class imbalance: problematic oranı {ratio:.1%}"
        )
    else:
        logger.info(
            f"Class balance kabul edilebilir: problematic oranı {ratio:.1%}"
        )


def run() -> pd.DataFrame:
    """
    Label engineering pipeline'ını çalıştırır.
    """
    logger.info("=" * 60)
    logger.info("ADIM 2: Label engineering başladı")
    logger.info("=" * 60)

    path = PROCESSED_FILES["order_level"]

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce data_loading.py çalıştırılmalı."
        )

    df = pd.read_parquet(path)

    logger.info(f"Order-level veri yüklendi -> {df.shape[0]:,} satır, {df.shape[1]} kolon")

    df = filter_modeling_orders(df)
    df = calculate_delivery_delay(df)
    df = create_target_label(df)
    df = add_customer_history(df)
    report_class_balance(df)

    output_path = PROCESSED_FILES["order_level_labeled"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("=" * 60)
    logger.info("ADIM 2 tamamlandı")
    logger.info(f"Kaydedildi -> {output_path}")
    logger.info(f"Final boyut: {df.shape[0]:,} satır, {df.shape[1]} kolon")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run()