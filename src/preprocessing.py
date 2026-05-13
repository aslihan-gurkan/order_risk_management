<<<<<<< HEAD
=======
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


>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
"""
Preprocessing modülü.

Amaç:
<<<<<<< HEAD
    Feature engineering sonrası oluşan featured.parquet dosyasını model eğitimine
    hazır hale getirmek.

Bu modülde:
    1. Featured veri okunur
    2. Temporal edge-period filtrelemesi yapılır
    3. Leakage, ID ve raw date kolonları çıkarılır
    4. Target / feature ayrımı yapılır
    5. Train / test split uygulanır
    6. Preprocessing pipeline sadece train set üzerinde fit edilir
    7. Train ve test setleri transform edilir
    8. Opsiyonel olarak SMOTE sadece train set üzerinde uygulanır
    9. Model input dosyaları ve preprocessor kaydedilir

Kritik prensipler:
    - Test datası preprocessing fit aşamasında görülmez.
    - SMOTE split öncesinde uygulanmaz.
    - Review, delivery delay, risk_score gibi post-order bilgiler modele verilmez.
    - Outlier'lar körü körüne silinmez; extreme transactional behavior business signal olabilir.
"""

from pathlib import Path
from typing import Dict, List, Tuple
=======
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
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

import joblib
import numpy as np
import pandas as pd
<<<<<<< HEAD
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    PROCESSED_FILES,
    MODELS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    APPLY_TEMPORAL_FILTER,
    APPLY_SMOTE,
    MODEL_FILES
)

from src.logger import get_logger
from src.utils.data_utils import grab_col_names
=======

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
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

logger = get_logger(__name__)


<<<<<<< HEAD
# ─────────────────────────────────────────────────────────────────────────────
# Genel ayarlar
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COL = "problematic_order"

MODEL_INPUT_PATH = Path(PROCESSED_FILES["train"]).parent / "model_input"
PREPROCESSOR_PATH = Path(MODEL_FILES["preprocessor"])



# Modelde kullanılmaması gereken kolonlar
ID_COLS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "main_seller_id",
    "product_id",
]

# Label üretiminde kullanılan veya sipariş tamamlandıktan sonra bilinen kolonlar
LEAKAGE_COLS = [
    "review_score",
    "review_creation_date",
    "review_comment_message",
    "review_comment_title",
    "review_missing",
    "delivery_delay_days",
    "is_undelivered",
    "label_low_review",
    "label_delivery_delay",
    "label_problematic_status",
    "risk_score",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_status",  # label sinyaliyle ilişkili olduğu için modelden çıkarılır
]

# Raw datetime kolonları feature engineering sonrası modele doğrudan verilmez.
# Bunlardan türetilen purchase_month, purchase_hour, estimated_delivery_days vb. kullanılabilir.
DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]

# Yüksek cardinality / operasyonel ID niteliğinde olup modele direkt verilmeyecek kolonlar
HIGH_CARDINALITY_DROP_COLS = [
    "customer_city",
]

# Varsa modelden çıkarılacak ekstra kolonlar
MANUAL_DROP_COLS = list(set(ID_COLS + LEAKAGE_COLS + DATE_COLS + HIGH_CARDINALITY_DROP_COLS))


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_dir() -> None:
    """Model input çıktı klasörünü oluşturur."""
    MODEL_INPUT_PATH.mkdir(parents=True, exist_ok=True)


def read_featured_data() -> pd.DataFrame:
    """Feature engineering çıktısı olan featured.parquet dosyasını okur."""
    featured_path = PROCESSED_FILES["featured"]

    if not Path(featured_path).exists():
        raise FileNotFoundError(
            "featured.parquet bulunamadı. Önce feature_engineering.py çalıştırılmalı."
        )

    df = pd.read_parquet(featured_path)
    logger.info(f"Featured veri yüklendi: {df.shape}")
    return df


def prepare_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tarih kolonlarını datetime formatına çevirir."""
    temp = df.copy()

    for col in DATE_COLS:
        if col in temp.columns:
            temp[col] = pd.to_datetime(temp[col], errors="coerce")

    return temp


def apply_temporal_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA'da görülen düşük hacimli edge-period ayları model training datasından çıkarır.

    Neden?
        2016 sonu ve 2018-09 sonrası dönemlerde problem rate uç değerlere çıkıyor.
        Bu büyük ihtimalle düşük sample size / incomplete period etkisi.
    """
    if not APPLY_TEMPORAL_FILTER:
        logger.info("Temporal filter kapalı. Tüm dönemler kullanılacak.")
        return df

    if "order_purchase_timestamp" not in df.columns:
        logger.warning("order_purchase_timestamp bulunamadı. Temporal filter uygulanamadı.")
        return df

    temp = df.copy()
    before_shape = temp.shape

    temp["order_purchase_timestamp"] = pd.to_datetime(
        temp["order_purchase_timestamp"],
        errors="coerce",
    )

    start_date = pd.to_datetime(TRAIN_START_DATE)
    end_date = pd.to_datetime(TRAIN_END_DATE)

    temp = temp[
        (temp["order_purchase_timestamp"] >= start_date)
        & (temp["order_purchase_timestamp"] < end_date)
    ].copy()

    logger.info(
        "Temporal filter uygulandı: "
        f"{TRAIN_START_DATE} <= order_purchase_timestamp < {TRAIN_END_DATE} | "
        f"önce={before_shape}, sonra={temp.shape}"
    )

    return temp


def drop_unusable_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Leakage, ID, raw date ve yüksek cardinality kolonları çıkarır.
    """
    existing_drop_cols = [col for col in MANUAL_DROP_COLS if col in df.columns]
    cleaned_df = df.drop(columns=existing_drop_cols)

    logger.info(f"Modelden çıkarılan kolon sayısı: {len(existing_drop_cols)}")
    logger.info(f"Modelden çıkarılan kolonlar: {existing_drop_cols}")

    return cleaned_df, existing_drop_cols


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Target ve feature ayrımı yapar."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target kolon bulunamadı: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y dağılımı:\n{y.value_counts(normalize=True).to_string()}")
=======
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
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

    return X, y


<<<<<<< HEAD
def identify_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Model input için kategorik ve numerik kolonları belirler.
    """
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)

    # cat_but_car içinden bazı kolonları özellikle tutmak isteyebiliriz.
    # main_product_category ve customer_state business olarak anlamlıdır ve OHE ile yönetilebilir.
    keep_high_card_cols = [
        col for col in ["main_product_category", "customer_state", "seller_state_mode"]
        if col in X.columns
    ]

    cat_cols = list(dict.fromkeys(cat_cols + keep_high_card_cols))

    # Gerçekten yüksek cardinality olan ve manuel drop listesinde olmayan kolonlar kaldıysa çıkarılır.
    # Bu güvenlik kontrolüdür.
    high_card_to_drop = [
        col for col in cat_but_car
        if col not in keep_high_card_cols
    ]

    if high_card_to_drop:
        logger.info(f"Ek yüksek cardinality kolonlar çıkarılıyor: {high_card_to_drop}")
        X.drop(columns=[col for col in high_card_to_drop if col in X.columns], inplace=True)
        cat_cols = [col for col in cat_cols if col not in high_card_to_drop]
        num_cols = [col for col in num_cols if col in X.columns]

    # Target yanlışlıkla kalmasın
    cat_cols = [col for col in cat_cols if col != TARGET_COL and col in X.columns]
    num_cols = [col for col in num_cols if col != TARGET_COL and col in X.columns]

    logger.info(f"Kategorik kolon sayısı: {len(cat_cols)}")
    logger.info(f"Numerik kolon sayısı: {len(num_cols)}")
    logger.info(f"Kategorik kolonlar: {cat_cols}")
    logger.info(f"Numerik kolonlar: {num_cols}")

    return cat_cols, num_cols


def create_one_hot_encoder() -> OneHotEncoder:
    """
    scikit-learn sürüm farklarına uyumlu OneHotEncoder oluşturur.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Numerik ve kategorik kolonlar için preprocessing pipeline oluşturur.

    Numerik:
        - Median imputation
        - RobustScaler

    Kategorik:
        - Most frequent imputation
        - OneHotEncoder
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )
=======
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
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744

    return preprocessor


<<<<<<< HEAD
def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    cat_cols: List[str],
    num_cols: List[str],
) -> List[str]:
    """Preprocessing sonrası oluşan feature isimlerini döndürür."""
    feature_names = []

    feature_names.extend(num_cols)

    if cat_cols:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = onehot.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(encoded_cat_names)

    return feature_names


def apply_smote_if_enabled(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> Tuple[np.ndarray, pd.Series]:
    """
    SMOTE'u sadece train set üzerinde uygular.

    Not:
        Test set üzerinde SMOTE uygulanmaz.
        Split öncesi SMOTE uygulanmaz.
    """
    if not APPLY_SMOTE:
        logger.info("SMOTE kapalı. Train set orijinal dağılımla kullanılacak.")
        return X_train, y_train

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE kullanmak için imbalanced-learn paketi gerekli. "
            "Kurulum: pip install imbalanced-learn"
        ) from exc

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info("SMOTE sadece train set üzerinde uygulandı.")
    logger.info(f"SMOTE öncesi train shape: {X_train.shape}")
    logger.info(f"SMOTE sonrası train shape: {X_resampled.shape}")
    logger.info(f"SMOTE sonrası y dağılımı:\n{pd.Series(y_resampled).value_counts(normalize=True).to_string()}")

    return X_resampled, pd.Series(y_resampled, name=TARGET_COL)


def save_preprocessing_outputs(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    metadata: Dict,
) -> None:
    """Preprocessing çıktılarını diske kaydeder."""
    ensure_output_dir()

    np.save(MODEL_INPUT_PATH / "X_train.npy", X_train)
    np.save(MODEL_INPUT_PATH / "X_test.npy", X_test)
    np.save(MODEL_INPUT_PATH / "y_train.npy", y_train.to_numpy())
    np.save(MODEL_INPUT_PATH / "y_test.npy", y_test.to_numpy())

    pd.Series(feature_names, name="feature_name").to_csv(
        MODEL_INPUT_PATH / "feature_names.csv",
        index=False,
    )

    pd.DataFrame([metadata]).to_csv(
        MODEL_INPUT_PATH / "preprocessing_metadata.csv",
        index=False,
    )

    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    logger.info(f"Preprocessing çıktıları kaydedildi: {MODEL_INPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Ana pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run() -> Dict:
    """Preprocessing pipeline'ını çalıştırır."""
    logger.info("=" * 60)
    logger.info("ADIM 5: Preprocessing başladı")
    logger.info("=" * 60)

    df = read_featured_data()
    df = prepare_datetime_columns(df)
    df = apply_temporal_filter(df)

    # Target boş olan kayıtlar varsa çıkarılır
    before_target_drop = df.shape
    df = df.dropna(subset=[TARGET_COL]).copy()
    logger.info(f"Target missing drop: önce={before_target_drop}, sonra={df.shape}")

    df, dropped_cols = drop_unusable_columns(df)

    X, y = split_features_target(df)
    cat_cols, num_cols = identify_feature_columns(X)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"X_train_raw shape: {X_train_raw.shape}")
    logger.info(f"X_test_raw shape: {X_test_raw.shape}")
    logger.info(f"y_train distribution:\n{y_train.value_counts(normalize=True).to_string()}")
    logger.info(f"y_test distribution:\n{y_test.value_counts(normalize=True).to_string()}")

    preprocessor = build_preprocessor(cat_cols, num_cols)

    # Fit sadece train üzerinde yapılır.
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    feature_names = get_transformed_feature_names(preprocessor, cat_cols, num_cols)

    X_train_final, y_train_final = apply_smote_if_enabled(X_train_processed, y_train)

    metadata = {
        "raw_rows_after_filter": len(df),
        "raw_feature_count": X.shape[1],
        "processed_train_rows": X_train_final.shape[0],
        "processed_test_rows": X_test_processed.shape[0],
        "processed_feature_count": X_train_final.shape[1],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "temporal_filter_applied": APPLY_TEMPORAL_FILTER,
        "train_start_date": TRAIN_START_DATE,
        "train_end_date_exclusive": TRAIN_END_DATE,
        "smote_applied": APPLY_SMOTE,
        "dropped_columns": ", ".join(dropped_cols),
        "categorical_columns": ", ".join(cat_cols),
        "numeric_columns": ", ".join(num_cols),
    }

    save_preprocessing_outputs(
        X_train=X_train_final,
        X_test=X_test_processed,
        y_train=y_train_final,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        metadata=metadata,
    )

    logger.info("=" * 60)
    logger.info("ADIM 5 tamamlandı")
    logger.info(f"Final X_train shape: {X_train_final.shape}")
    logger.info(f"Final X_test shape: {X_test_processed.shape}")
    logger.info("=" * 60)

    return metadata


if __name__ == "__main__":
    run()
=======
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
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
