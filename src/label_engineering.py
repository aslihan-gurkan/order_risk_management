"""
Label engineering modülü.

Amaç:
    Olist veri setinde gerçek "iade" label'ı olmadığı için
    iş problemiyle uyumlu, açıklanabilir bir binary hedef değişken üretmek.

Problem:
    Sipariş tamamlanmadan önce sorunlu sipariş riskini tahmin etmek.

Hedef değişken:
    problematic_order
        0 = normal sipariş
        1 = sorunlu sipariş

Kritik prensip:
    Review score ve gerçek teslimat bilgisi label üretiminde kullanılabilir.
    Ancak model feature'ı olarak kullanılmaz.
    Aksi halde data leakage oluşur.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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

    shipped, invoiced, processing gibi statülerde sipariş sonucu belirsizdir.
    Bu satırlara güvenilir label veremeyiz.
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

    Teslim edilmeyen siparişlerde delivery_delay_days 999 yapılır.
    Bu bilgi sadece label üretiminde kullanılır, model feature'ı olmaz.
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
    Binary hedef değişkeni risk_score mantığıyla oluşturur.

    Neden eski OR mantığını değiştirdik?
        Eski yaklaşımda review_score <= 2 tek başına problematic_order=1 yapıyordu.
        Bu da target'ı düşük review sinyalinin domine etmesine ve noisy label'a neden oluyordu.

    Yeni yaklaşım:
        - Çok ciddi status problemi güçlü sinyal
        - Ciddi teslimat gecikmesi güçlü sinyal
        - Düşük review yardımcı sinyal
        - Çok düşük review daha güçlü yardımcı sinyal

    Not:
        freight_ratio, seller_count gibi model feature'ı olacak alanları label üretiminde
        kullanmıyoruz. Böylece modelin target kuralını doğrudan öğrenmesini engelliyoruz.
    """
    review_score = df["review_score"].fillna(3)

    # Açıklanabilir label sinyal kolonları
    df["label_low_review"] = (
        review_score <= DISSATISFACTION_REVIEW_THRESHOLD
    ).astype(int)

    df["label_very_low_review"] = (
        review_score <= 1
    ).astype(int)

    df["label_delivery_delay"] = (
        (df["is_undelivered"] == 0) &
        (df["delivery_delay_days"] > DELIVERY_DELAY_THRESHOLD)
<<<<<<< HEAD
    ).astype(int)

    df["label_severe_delivery_delay"] = (
        (df["is_undelivered"] == 0) &
        (df["delivery_delay_days"] > 15)
    ).astype(int)
=======
    ).astype(int) 
    # Teslim edildiyse ve geç kaldıysa -> delivery delay
    # Teslim edilmediyse -> status problem
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

    df["label_problematic_status"] = (
        df["order_status"].isin(PROBLEMATIC_ORDER_STATUSES)
    ).astype(int)

    # Risk score
    df["risk_score"] = 0

    # Review yardımcı sinyal: tek başına target'ı domine etmesin
    df.loc[df["label_low_review"] == 1, "risk_score"] += 1
    df.loc[df["label_very_low_review"] == 1, "risk_score"] += 1

    # Teslimat gecikmesi daha operasyonel ve güçlü sinyal
    df.loc[df["label_delivery_delay"] == 1, "risk_score"] += 2
    df.loc[df["label_severe_delivery_delay"] == 1, "risk_score"] += 1

    # Canceled / unavailable en güçlü sinyal
    df.loc[df["label_problematic_status"] == 1, "risk_score"] += 4

    # Final target
    # 3 ve üzeri:
    #   - ciddi gecikme
    #   - status problemi
    #   - çok düşük review + başka sinyal kombinasyonu
    # gibi daha anlamlı problematic order'ları yakalar.
    df["problematic_order"] = (df["risk_score"] >= 3).astype(int)

    logger.info("Risk score dağılımı:")
    logger.info(df["risk_score"].value_counts().sort_index().to_string())

    total = len(df)
    problematic = df["problematic_order"].sum()
    normal = total - problematic

    logger.info("Hedef değişken dağılımı:")
    logger.info(f"  Normal      (0): {normal:,} ({normal / total:.1%})")
    logger.info(f"  Problematic (1): {problematic:,} ({problematic / total:.1%})")

    logger.info("Label sinyalleri:")
    logger.info(f"  Düşük review          : {df['label_low_review'].sum():,}")
    logger.info(f"  Çok düşük review      : {df['label_very_low_review'].sum():,}")
    logger.info(f"  Teslimat gecikmesi    : {df['label_delivery_delay'].sum():,}")
    logger.info(f"  Ciddi teslimat gecikmesi: {df['label_severe_delivery_delay'].sum():,}")
    logger.info(f"  Status problemi       : {df['label_problematic_status'].sum():,}")

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
    """
    total = len(df)
    problematic = df["problematic_order"].sum()
    ratio = problematic / total

    if ratio < 0.10:
        logger.warning(f"Ciddi class imbalance: problematic oranı {ratio:.1%}")
    elif ratio < 0.30:
        logger.warning(f"Hafif class imbalance: problematic oranı {ratio:.1%}")
    else:
        logger.info(f"Class balance kabul edilebilir: problematic oranı {ratio:.1%}")


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

<<<<<<< HEAD
    logger.info(
        f"Order-level veri yüklendi -> {df.shape[0]:,} satır, {df.shape[1]} kolon"
    )
=======
    logger.info(f"Order-level veri yüklendi -> {df.shape[0]:,} satır, {df.shape[1]} kolon")
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

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