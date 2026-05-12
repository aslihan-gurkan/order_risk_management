#4

# 1. Temporal Split - tarihe Göre
# İlk %80 zaman -> train
# Son %20 zaman -> test

# 2. Feature / target ayrımı
# X = features
# y = problematic_order

# Kolon Tipleri Ayırma (Numerical/Categorical)
# N: price, freight_ratio, approval_delay_hours, estimated_delivery_days, seller_historical_problem_rate, etc.
# C: payment_type, customer_state, seller_state_mode, main_product_category, etc.

# Low cardinality categorical: OneHotEncoder
# High cardinality categorical: TargetEncoder => main_seller_id, main_product_category, customer_city

# problematic rate = %14.5


"""
Preprocessing modülü.

Amaç:
    Feature engineering sonrası oluşan veriyi
    modellemeye hazır hale getirmek.

Bu modülde:
    1. Veri yüklenir
    2. Temporal filtre uygulanır
    3. Missing değerler business-aware şekilde doldurulur
    4. Train / test temporal split yapılır
    5. Leakage ve ID kolonları çıkarılır
    6. Numerical / categorical kolonlar ayrılır
    7. sklearn preprocessing pipeline oluşturulur
    8. Train / test parquet kaydedilir
    9. Preprocessing pipeline kaydedilir

Önemli:
    Random split kullanılmaz.
    Çünkü problem zaman bazlı tahmin problemidir.

Production yaklaşımı:
    Geçmiş siparişlerden öğren ->
    gelecekteki siparişleri tahmin et
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from category_encoders.target_encoder import TargetEncoder

from src.config import (
    PROCESSED_FILES,
    MODEL_FILES
)


from src.logger import get_logger

logger = get_logger(__name__)


# ── Sabitler ──────────────────────────────────────────────────────────────────

from src.feature_config import (
    DROP_COLUMNS,
    HIGH_CARDINALITY_COLS,
    LOW_CARDINALITY_COLS,
    TARGET_COLUMN,
)

# ── Veri filtreleme ───────────────────────────────────────────────────────────
def apply_temporal_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anomalik ve incomplete dönemleri çıkarır.

    Neden?
        2016 başı -> çok düşük hacimli anomalik kayıtlar
        2018-09 sonrası -> incomplete period

    Bu filtre temporal consistency sağlar.
    """
    before = len(df)

    df = df[
        (df["order_purchase_timestamp"] >= "2017-01-01") &
        (df["order_purchase_timestamp"] < "2018-09-01")
    ].copy()

    after = len(df)

    logger.info(
        f"Temporal filtre uygulandı -> "
        f"{before - after:,} satır çıkarıldı, "
        f"{after:,} satır kaldı"
    )

    return df


# ── Missing value işlemleri ───────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing değerleri business-aware şekilde doldurur.

    Kritik yaklaşım:
        Null gördüğümüz her satırı silmiyoruz.
        Önce neden oluştuğunu anlamaya çalışıyoruz.

    Örnek:
        canceled/unavailable siparişlerde
        item/seller bilgisi olmayabilir.
    """

    # ── Numerical business null'ları ────────────────────────────────────────
    zero_fill_cols = [
        "item_count",
        "product_count",
        "seller_count",
        "total_price",
        "total_freight_value",
        "seller_historical_problem_count",
        "seller_historical_order_count",
    ]

    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Seller bilgisi olmayanlar ───────────────────────────────────────────
    if "main_seller_id" in df.columns:
        df["main_seller_id"] = df["main_seller_id"].fillna(
            "unknown_seller"
        )

    if "seller_state_mode" in df.columns:
        df["seller_state_mode"] = df["seller_state_mode"].fillna(
            "unknown"
        )

    # ── Payment bilgisi ─────────────────────────────────────────────────────
    if "payment_type" in df.columns:
        df["payment_type"] = df["payment_type"].fillna(
            "unknown"
        )

    # ── Approval delay eksikleri ────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    logger.info("Missing value işlemleri tamamlandı")

    return df


# ── Temporal split ────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame):
    """
    Zamansal train/test split uygular.

    Train:
        2017-01 -> 2018-05

    Test:
        2018-06 -> 2018-08

    Neden random split kullanmıyoruz?
        Çünkü gerçek problem:
            geçmişten öğren ->
            geleceği tahmin et
    """

    train_df = df[
        df["order_purchase_timestamp"] < "2018-06-01"
    ].copy()

    test_df = df[
        df["order_purchase_timestamp"] >= "2018-06-01"
    ].copy()

    logger.info(
        f"Temporal split tamamlandı -> "
        f"train: {train_df.shape}, "
        f"test: {test_df.shape}"
    )

    logger.info(
        f"Train problematic oranı: "
        f"{train_df[TARGET_COLUMN].mean():.2%}"
    )

    logger.info(
        f"Test problematic oranı: "
        f"{test_df[TARGET_COLUMN].mean():.2%}"
    )

    return train_df, test_df


# ── Feature / target ayırma ──────────────────────────────────────────────────
def split_features_target(df: pd.DataFrame):
    """
    Feature ve target kolonlarını ayırır.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# ── Preprocessing pipeline ────
def build_preprocessor(X: pd.DataFrame):
    """
    sklearn preprocessing pipeline oluşturur.

    Pipeline avantajı:
        - train/test consistency
        - inference consistency
        - production-ready yapı
    """

    # ── Kategorik kolonları belirle ────
    categorical_cols = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # High-cardinality kolonlar
    high_cardinality_cols = [
        col for col in HIGH_CARDINALITY_COLS
        if col in categorical_cols
    ]

    # Low-cardinality kolonlar
    low_cardinality_cols = [
        col for col in LOW_CARDINALITY_COLS
        if col in categorical_cols
    ]

    # Numerical kolonlar
    numerical_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    logger.info(f"Numerical kolon sayısı: {len(numerical_cols)}")
    logger.info(f"Low-cardinality kolonlar: {low_cardinality_cols}")
    logger.info(f"High-cardinality kolonlar: {high_cardinality_cols}")

    # ── Numerical pipeline ────────────────────────────────────────────────
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # ── Low-cardinality categorical ───────────────────────────────────────
    low_cardinality_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ── High-cardinality categorical ──────────────────────────────────────
    high_cardinality_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder()),
    ])

    # ── ColumnTransformer ─────────────────────────────────────────────────
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_cols
        ),
        (
            "low_cardinality",
            low_cardinality_pipeline,
            low_cardinality_cols
        ),
        (
            "high_cardinality",
            high_cardinality_pipeline,
            high_cardinality_cols
        ),
    ])

    logger.info("Preprocessing pipeline oluşturuldu")

    return preprocessor


# ── Ana pipeline ────
def run():
    """
    Tüm preprocessing pipeline'ını çalıştırır.
    """

    logger.info("=" * 60)
    logger.info("ADIM 4: Preprocessing başladı")
    logger.info("=" * 60)

    # ── Veri yükleme ─────
    path = PROCESSED_FILES["featured"]

    if not Path(path).exists():
        raise FileNotFoundError(
            "Önce feature_engineering.py çalıştırılmalı."
        )

    df = pd.read_parquet(path)

    logger.info(
        f"Featured veri yüklendi -> "
        f"{df.shape[0]:,} satır, {df.shape[1]} kolon"
    )

    # ── Temporal filtre ───────────────────────────────────────────────────
    df = apply_temporal_filters(df)

    # ── Missing value işlemleri ───────────────────────────────────────────
    df = handle_missing_values(df)

    # ── Temporal split ────────────────────────────────────────────────────
    train_df, test_df = temporal_split(df)

    # ── Train/test kaydetme ───────────────────────────────────────────────
    train_path = PROCESSED_FILES["train"]
    test_path = PROCESSED_FILES["test"]

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Train kaydedildi -> {train_path}")
    logger.info(f"Test kaydedildi -> {test_path}")

    # ── Feature / target ayrımı ───────────────────────────────────────────
    X_train, y_train = split_features_target(train_df)

    # ── ID kolonlarını çıkar ──────────────────────────────────────────────
    existing_drop_cols = [
        col for col in DROP_COLUMNS
        if col in X_train.columns
    ]

    X_train = X_train.drop(columns=existing_drop_cols)

    logger.info(
        f"Drop edilen kolonlar: {existing_drop_cols}"
    )

    # ── Preprocessor oluştur ──────────────────────────────────────────────
    preprocessor = build_preprocessor(X_train)

    # ── Fit ───────────────────────────────────────────────────────────────
    preprocessor.fit(X_train, y_train)

    logger.info("Preprocessor fit edildi")

    # ── Kaydet ────────────────────────────────────────────────────────────
    output_path = MODEL_FILES["preprocessor"]

    joblib.dump(preprocessor, output_path)

    logger.info(f"Preprocessor kaydedildi -> {output_path}")

    logger.info("=" * 60)
    logger.info("ADIM 4 tamamlandı ")
    logger.info("=" * 60)

    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
    }


if __name__ == "__main__":
    run()