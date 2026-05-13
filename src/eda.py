"""
EDA modülü.

Amaç:
    Modelleme öncesinde veri setini, hedef değişkeni, veri kalitesini,
    değişken tiplerini, outlier yapılarını ve temel iş sinyallerini analiz etmek.

Bu modülde:
    1. Labeled ve featured veri okunur
    2. Miuul tarzı değişken tipleri çıkarılır
    3. Missing value analizi yapılır
    4. Outlier analizi yapılır
    5. Target dağılımı incelenir
    6. Label sinyalleri analiz edilir
    7. Zaman bazlı problematic order trendi çıkarılır
    8. Kategori, ödeme tipi, eyalet ve seller risk analizleri yapılır
    9. Sunumda kullanılabilecek CSV ve PNG çıktıları üretilir

Kritik not:
    EDA model eğitmez.
    Amaç veriyi anlamak, feature engineering kararlarını kanıtlamak
    ve sunum/storytelling için görsel destek üretmektir.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_FILES, OUTPUTS_PATH
from src.logger import get_logger

<<<<<<< HEAD
from src.utils.data_utils import (
    grab_col_names,
    missing_values_table,
    outlier_thresholds,
    check_outlier,
    target_summary_with_cat,
    target_summary_with_num,
    rare_analyser,
)

=======
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Genel ayarlar
# ─────────────────────────────────────────────────────────────────────────────
EDA_OUTPUT_PATH = Path(OUTPUTS_PATH) / "eda"
EDA_PLOTS_PATH = EDA_OUTPUT_PATH / "plots"

TARGET_COL = "problematic_order"
ID_COLS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "main_seller_id",
    "product_id",
]

LEAKAGE_COLS = [
    "review_score",
    "review_comment_message",
    "review_comment_title",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "delivery_delay_days",
    "label_low_review",
    "label_delivery_delay",
    "label_problematic_status",
    "risk_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_dirs() -> None:
    """EDA çıktı klasörlerini oluşturur."""
    EDA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    EDA_PLOTS_PATH.mkdir(parents=True, exist_ok=True)


def save_output(df: pd.DataFrame, filename: str) -> None:
    """EDA sonucunu data/outputs/eda klasörüne kaydeder."""
    output_path = EDA_OUTPUT_PATH / filename
    df.to_csv(output_path, index=False)
    logger.info(f"EDA çıktısı kaydedildi: {output_path}")


def save_plot(filename: str) -> None:
    """Aktif matplotlib figürünü kaydeder."""
    output_path = EDA_PLOTS_PATH / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"EDA görseli kaydedildi: {output_path}")


def safe_rate(series: pd.Series) -> float:
    """Boş serilerde hata almamak için güvenli oran hesaplar."""
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def prepare_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp kolonlarını datetime formatına çevirir."""
    temp = df.copy()

    date_cols = [col for col in temp.columns if "date" in col or "timestamp" in col]
    for col in date_cols:
        temp[col] = pd.to_datetime(temp[col], errors="coerce")

    return temp

<<<<<<< HEAD
=======

def grab_col_names(
    dataframe: pd.DataFrame,
    cat_th: int = 10,
    car_th: int = 20,
) -> Tuple[List[str], List[str], List[str], List[str]]:

    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype == "O"
        or pd.api.types.is_string_dtype(dataframe[col])
        or pd.api.types.is_categorical_dtype(dataframe[col])
    ]

    num_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    num_but_cat = [
        col for col in num_cols
        if dataframe[col].nunique(dropna=True) < cat_th
    ]

    cat_but_car = [
        col for col in cat_cols
        if dataframe[col].nunique(dropna=True) > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car, num_but_cat

def missing_values_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Eksik değer sayısı ve oranını döndürür."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    result = pd.DataFrame({
        "variable": na_columns,
        "missing_count": [dataframe[col].isnull().sum() for col in na_columns],
        "missing_ratio": [dataframe[col].isnull().mean() for col in na_columns],
    }).sort_values("missing_ratio", ascending=False)

    return result


def outlier_thresholds(
    dataframe: pd.DataFrame,
    col_name: str,
    q1: float = 0.05,
    q3: float = 0.95,
) -> Tuple[float, float]:
    """IQR tabanlı alt ve üst outlier eşiklerini döndürür."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outlier(dataframe: pd.DataFrame, col_name: str) -> bool:
    """Bir numerik değişkende outlier var mı kontrol eder."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    return outliers.any(axis=None)


>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
def outlier_summary(dataframe: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """Numerik değişkenler için outlier özet tablosu üretir."""
    rows = []

    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            continue
        if col in ID_COLS or col == TARGET_COL:
            continue

        series = dataframe[col].dropna()
        if series.empty:
            continue

        low_limit, up_limit = outlier_thresholds(dataframe, col)
        outlier_mask = (dataframe[col] < low_limit) | (dataframe[col] > up_limit)

        rows.append({
            "variable": col,
            "low_limit": low_limit,
            "up_limit": up_limit,
            "outlier_count": int(outlier_mask.sum()),
            "outlier_ratio": float(outlier_mask.mean()),
            "min": dataframe[col].min(),
            "max": dataframe[col].max(),
            "mean": dataframe[col].mean(),
            "median": dataframe[col].median(),
        })

    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False)


<<<<<<< HEAD
=======
def target_summary_with_cat(
    dataframe: pd.DataFrame,
    target: str,
    categorical_col: str,
    min_count: int = 50,
) -> pd.DataFrame:
    """Kategorik değişken kırılımında target oranını döndürür."""
    result = (
        dataframe
        .groupby(categorical_col, dropna=False)
        .agg(
            order_count=(target, "count"),
            problematic_count=(target, "sum"),
            problematic_rate=(target, "mean"),
        )
        .reset_index()
        .query("order_count >= @min_count")
        .sort_values("problematic_rate", ascending=False)
    )

    return result


def target_summary_with_num(
    dataframe: pd.DataFrame,
    target: str,
    numerical_col: str,
) -> pd.DataFrame:
    """Target kırılımında numerik değişken özetini döndürür."""
    result = (
        dataframe
        .groupby(target)
        .agg(
            count=(numerical_col, "count"),
            mean=(numerical_col, "mean"),
            median=(numerical_col, "median"),
            min=(numerical_col, "min"),
            max=(numerical_col, "max"),
        )
        .reset_index()
    )

    result.insert(0, "variable", numerical_col)
    return result


def rare_analyser(
    dataframe: pd.DataFrame,
    target: str,
    cat_cols: List[str],
    max_unique: int = 50,
) -> pd.DataFrame:
    """Kategorik değişkenlerde rare sınıf ve target oranı analizi yapar."""
    rows = []

    for col in cat_cols:
        if col == target or dataframe[col].nunique(dropna=True) > max_unique:
            continue

        summary = (
            dataframe
            .groupby(col, dropna=False)
            .agg(
                count=(target, "count"),
                ratio=(target, lambda x: len(x) / len(dataframe)),
                target_mean=(target, "mean"),
            )
            .reset_index()
        )

        summary.insert(0, "variable", col)
        summary = summary.rename(columns={col: "category"})
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
# ─────────────────────────────────────────────────────────────────────────────
# Veri kalite analizleri
# ─────────────────────────────────────────────────────────────────────────────
def analyze_variable_types(df: pd.DataFrame) -> Dict[str, int]:
    """cat_cols, num_cols, cat_but_car, num_but_cat çıktılarını üretir."""
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

    result = pd.DataFrame({
        "group": ["cat_cols", "num_cols", "cat_but_car", "num_but_cat"],
        "count": [len(cat_cols), len(num_cols), len(cat_but_car), len(num_but_cat)],
        "columns": [
            ", ".join(cat_cols),
            ", ".join(num_cols),
            ", ".join(cat_but_car),
            ", ".join(num_but_cat),
        ],
    })

    save_output(result, "eda_variable_types.csv")

    return {
        "cat_cols": len(cat_cols),
        "num_cols": len(num_cols),
        "cat_but_car": len(cat_but_car),
        "num_but_cat": len(num_but_cat),
    }


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik değer analizi üretir."""
    result = missing_values_table(df)
    save_output(result, "eda_missing_values.csv")

    if not result.empty:
        plot_df = result.head(20).sort_values("missing_ratio")
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["variable"], plot_df["missing_ratio"])
        plt.title("Top Missing Value Ratios")
        plt.xlabel("Missing Ratio")
        plt.ylabel("Variable")
        save_plot("missing_values_top20.png")

    return result


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Outlier analizi üretir."""
    _, num_cols, _, _ = grab_col_names(df)
    result = outlier_summary(df, num_cols)
    save_output(result, "eda_outlier_summary.csv")

    if not result.empty:
        plot_df = result.head(20).sort_values("outlier_ratio")
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["variable"], plot_df["outlier_ratio"])
        plt.title("Top Outlier Ratios")
        plt.xlabel("Outlier Ratio")
        plt.ylabel("Variable")
        save_plot("outlier_ratios_top20.png")

    return result


def analyze_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature olarak kullanılmaması gereken leakage kolonlarını raporlar.
    Bu tablo sunumda özellikle önemlidir.
    """
    rows = []

    for col in LEAKAGE_COLS:
        rows.append({
            "column": col,
            "exists_in_dataset": col in df.columns,
            "reason": "Post-order / label-generation information; should not be used as prediction feature.",
        })

    result = pd.DataFrame(rows)
    save_output(result, "eda_leakage_columns.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Target ve label analizleri
# ─────────────────────────────────────────────────────────────────────────────
def analyze_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """problematic_order hedef değişkeninin dağılımını çıkarır."""
    result = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
    )

    result["ratio"] = result["count"] / result["count"].sum()
    save_output(result, "eda_target_distribution.csv")

    plt.figure(figsize=(7, 5))
    plt.bar(result[TARGET_COL].astype(str), result["count"])
    plt.title("Target Distribution: Problematic Order")
    plt.xlabel("Problematic Order")
    plt.ylabel("Order Count")
    save_plot("target_distribution.png")

    return result


def analyze_label_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Problematic order label'ının hangi sinyallerden oluştuğunu gösterir.
    """
    signal_cols = [
        "label_low_review",
        "label_delivery_delay",
        "label_problematic_status",
    ]

    existing_cols = [col for col in signal_cols if col in df.columns]

    result = pd.DataFrame({
        "signal": existing_cols,
        "count": [df[col].sum() for col in existing_cols],
    })

    if result.empty:
        save_output(result, "eda_label_signals.csv")
        return result

    result["ratio_in_all_orders"] = result["count"] / len(df)
    save_output(result, "eda_label_signals.csv")

    plot_df = result.sort_values("count")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["signal"], plot_df["count"])
    plt.title("Label Engineering Signal Counts")
    plt.xlabel("Order Count")
    plt.ylabel("Signal")
    save_plot("label_signal_counts.png")

    return result


def analyze_monthly_problem_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Aylık problematic order oranını hesaplar."""
    if "order_purchase_timestamp" not in df.columns:
        return pd.DataFrame()

    temp = prepare_datetime_columns(df)
    temp = temp.dropna(subset=["order_purchase_timestamp"])

    temp["purchase_month_period"] = (
        temp["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        temp
        .groupby("purchase_month_period", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            problematic_count=(TARGET_COL, "sum"),
            problematic_rate=(TARGET_COL, "mean"),
        )
    )

    save_output(result, "eda_monthly_problem_rate.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(result["purchase_month_period"], result["problematic_rate"], marker="o")
    plt.title("Monthly Problematic Order Rate")
    plt.xlabel("Purchase Month")
    plt.ylabel("Problematic Rate")
    plt.xticks(rotation=45)
    save_plot("monthly_problematic_rate.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Business kırılım analizleri
# ─────────────────────────────────────────────────────────────────────────────
def analyze_category_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Ürün kategorisi bazında problematic order oranını hesaplar."""
    if "main_product_category" not in df.columns:
        return pd.DataFrame()

    result = target_summary_with_cat(
        dataframe=df,
        target=TARGET_COL,
        categorical_col="main_product_category",
        min_count=100,
    )

    save_output(result, "eda_category_risk.csv")

    if not result.empty:
        plot_df = result.head(15).sort_values("problematic_rate")
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["main_product_category"].astype(str), plot_df["problematic_rate"])
        plt.title("Top Risky Categories")
        plt.xlabel("Problematic Rate")
        plt.ylabel("Product Category")
        save_plot("top_risky_categories.png")

    return result


def analyze_payment_type_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Ödeme tipi bazında problematic order oranını hesaplar."""
    if "payment_type" not in df.columns:
        return pd.DataFrame()

    result = target_summary_with_cat(
        dataframe=df,
        target=TARGET_COL,
        categorical_col="payment_type",
        min_count=10,
    )

    save_output(result, "eda_payment_type_risk.csv")

    if not result.empty:
        plot_df = result.sort_values("problematic_rate")
        plt.figure(figsize=(8, 5))
        plt.barh(plot_df["payment_type"].astype(str), plot_df["problematic_rate"])
        plt.title("Payment Type Risk")
        plt.xlabel("Problematic Rate")
        plt.ylabel("Payment Type")
        save_plot("payment_type_risk.png")

    return result


def analyze_customer_state_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Müşteri eyaleti bazında problematic order oranını hesaplar."""
    if "customer_state" not in df.columns:
        return pd.DataFrame()

    result = target_summary_with_cat(
        dataframe=df,
        target=TARGET_COL,
        categorical_col="customer_state",
        min_count=100,
    )

    save_output(result, "eda_customer_state_risk.csv")

    if not result.empty:
        plot_df = result.head(15).sort_values("problematic_rate")
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["customer_state"].astype(str), plot_df["problematic_rate"])
        plt.title("Top Risky Customer States")
        plt.xlabel("Problematic Rate")
        plt.ylabel("Customer State")
        save_plot("top_risky_customer_states.png")

    return result


def analyze_seller_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seller historical problem rate dağılımını inceler.

    main_seller_id modele direkt verilmez.
    Ancak EDA ve operasyonel raporlama için kullanılabilir.
    """
    if "main_seller_id" not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        "order_count": ("order_id", "nunique"),
        "problematic_count": (TARGET_COL, "sum"),
        "problematic_rate": (TARGET_COL, "mean"),
    }

    if "seller_historical_problem_rate" in df.columns:
        agg_dict["avg_historical_problem_rate"] = ("seller_historical_problem_rate", "mean")

    result = (
        df
        .groupby("main_seller_id", as_index=False)
        .agg(**agg_dict)
        .query("order_count >= 20")
        .sort_values("problematic_rate", ascending=False)
    )

    save_output(result, "eda_seller_risk.csv")

    if not result.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(result["problematic_rate"].dropna(), bins=30)
        plt.title("Seller Problematic Rate Distribution")
        plt.xlabel("Seller Problematic Rate")
        plt.ylabel("Seller Count")
        save_plot("seller_problematic_rate_distribution.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature odaklı EDA analizleri
# ─────────────────────────────────────────────────────────────────────────────
def analyze_freight_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Freight ratio dağılımını ve target ilişkisini analiz eder."""
    if "freight_ratio" not in df.columns:
        return pd.DataFrame()

    result = target_summary_with_num(df, TARGET_COL, "freight_ratio")
    save_output(result, "eda_freight_ratio_by_target.csv")

    plt.figure(figsize=(8, 5))
    plt.hist(df["freight_ratio"].dropna(), bins=50)
    plt.title("Freight Ratio Distribution")
    plt.xlabel("Freight Ratio")
    plt.ylabel("Order Count")
    save_plot("freight_ratio_distribution.png")

    return result


def analyze_shipping_limit_days(df: pd.DataFrame) -> pd.DataFrame:
    """Shipping limit days değişkeninin target ile ilişkisini analiz eder."""
    if "shipping_limit_days" not in df.columns:
        return pd.DataFrame()

    result = target_summary_with_num(df, TARGET_COL, "shipping_limit_days")
    save_output(result, "eda_shipping_limit_days_by_target.csv")

    plt.figure(figsize=(8, 5))
    plt.hist(df["shipping_limit_days"].dropna(), bins=40)
    plt.title("Shipping Limit Days Distribution")
    plt.xlabel("Shipping Limit Days")
    plt.ylabel("Order Count")
    save_plot("shipping_limit_days_distribution.png")

    return result


def analyze_delay_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    delivery_delay_days post-order bilgi olduğu için modele feature olarak verilmez.
    Ancak label kalitesini ve iş problemini anlatmak için EDA'da analiz edilir.
    """
    if "delivery_delay_days" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp["delivery_delay_bucket"] = pd.cut(
        temp["delivery_delay_days"],
        bins=[-np.inf, 0, 3, 7, 14, np.inf],
        labels=["no_delay", "1_3_days", "4_7_days", "8_14_days", "15_plus_days"],
    )

    result = target_summary_with_cat(
        dataframe=temp,
        target=TARGET_COL,
        categorical_col="delivery_delay_bucket",
        min_count=10,
    )

    save_output(result, "eda_delivery_delay_bucket_risk.csv")

    if not result.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(result["delivery_delay_bucket"].astype(str), result["problematic_rate"])
        plt.title("Delivery Delay Bucket Risk")
        plt.xlabel("Delivery Delay Bucket")
        plt.ylabel("Problematic Rate")
        plt.xticks(rotation=30)
        save_plot("delivery_delay_bucket_risk.png")

    return result


def analyze_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numerik değişkenler için correlation matrix üretir."""
    _, num_cols, _, _ = grab_col_names(df)
    selected_num_cols = [
        col
        for col in num_cols
        if col not in ID_COLS
        and col not in LEAKAGE_COLS
        and df[col].nunique(dropna=True) > 2
    ]

    if TARGET_COL in df.columns:
        selected_num_cols = selected_num_cols + [TARGET_COL]

    if len(selected_num_cols) < 2:
        return pd.DataFrame()

    corr = df[selected_num_cols].corr(numeric_only=True)
    corr_output = corr.reset_index().rename(columns={"index": "variable"})
    save_output(corr_output, "eda_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    save_plot("correlation_matrix.png")

    return corr_output


def analyze_target_with_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Önemli numerik değişkenlerin target kırılımında özetini çıkarır."""
    candidate_cols = [
        "freight_ratio",
        "shipping_limit_days",
        "seller_historical_problem_rate",
        "category_historical_problem_rate",
        "payment_value",
        "freight_value",
        "product_weight_g",
        "product_volume_cm3",
        "seller_order_count",
        "customer_order_count",
    ]

    existing_cols = [col for col in candidate_cols if col in df.columns]
    summaries = [target_summary_with_num(df, TARGET_COL, col) for col in existing_cols]

    if not summaries:
        return pd.DataFrame()

    result = pd.concat(summaries, ignore_index=True)
    save_output(result, "eda_numeric_features_by_target.csv")
    return result


def analyze_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Rare kategorik sınıf analizi yapar."""
    cat_cols, _, _, _ = grab_col_names(df)
    excluded = set(ID_COLS + [TARGET_COL])
    selected_cat_cols = [col for col in cat_cols if col not in excluded]

    result = rare_analyser(df, TARGET_COL, selected_cat_cols)
    save_output(result, "eda_rare_categories.csv")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ana pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run() -> dict:
    """EDA pipeline'ını çalıştırır."""
    logger.info("=" * 60)
<<<<<<< HEAD
    logger.info("ADIM 4: EDA başladı")
=======
    logger.info("ADIM 3A: EDA başladı")
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
    logger.info("=" * 60)

    ensure_output_dirs()

    labeled_path = PROCESSED_FILES["order_level_labeled"]
    featured_path = PROCESSED_FILES["featured"]

    if not Path(labeled_path).exists():
        raise FileNotFoundError(
            "order_level_labeled bulunamadı. Önce label_engineering çalıştırılmalı."
        )

    if not Path(featured_path).exists():
        raise FileNotFoundError(
            "featured bulunamadı. Önce feature_engineering çalıştırılmalı."
        )

    labeled_df = pd.read_parquet(labeled_path)
    featured_df = pd.read_parquet(featured_path)

    labeled_df = prepare_datetime_columns(labeled_df)
    featured_df = prepare_datetime_columns(featured_df)

    logger.info(f"Labeled veri yüklendi: {labeled_df.shape}")
    logger.info(f"Featured veri yüklendi: {featured_df.shape}")

    # Veri kalite analizleri
    variable_type_summary = analyze_variable_types(featured_df)
    missing_values = analyze_missing_values(featured_df)
    outliers = analyze_outliers(featured_df)
    leakage_columns = analyze_leakage_columns(featured_df)

    # Target ve label analizleri
    target_dist = analyze_target_distribution(labeled_df)
    label_signals = analyze_label_signals(labeled_df)
    monthly_rate = analyze_monthly_problem_rate(labeled_df)

    # Business kırılımları
    category_risk = analyze_category_risk(featured_df)
    payment_risk = analyze_payment_type_risk(featured_df)
    state_risk = analyze_customer_state_risk(featured_df)
    seller_risk = analyze_seller_risk(featured_df)

    # Feature odaklı analizler
    freight_ratio = analyze_freight_ratio(featured_df)
    shipping_limit_days = analyze_shipping_limit_days(featured_df)
    delay_buckets = analyze_delay_buckets(labeled_df)
    correlation_matrix = analyze_correlation_matrix(featured_df)
    numeric_target_summary = analyze_target_with_numeric_features(featured_df)
    rare_categories = analyze_rare_categories(featured_df)

    logger.info("Target dağılımı:")
    logger.info(target_dist.to_string(index=False))

    logger.info("Label sinyal özeti:")
    logger.info(label_signals.to_string(index=False))

    logger.info("Değişken tip özeti:")
    logger.info(variable_type_summary)

    logger.info("=" * 60)
<<<<<<< HEAD
    logger.info("ADIM 4 tamamlandı")
=======
    logger.info("ADIM 3A tamamlandı")
>>>>>>> b76eb810bfb118acd8ca344fa7242c89374e1744
    logger.info("=" * 60)

    return {
        "target_distribution_rows": len(target_dist),
        "label_signal_rows": len(label_signals),
        "monthly_rows": len(monthly_rate),
        "category_rows": len(category_risk),
        "payment_rows": len(payment_risk),
        "state_rows": len(state_risk),
        "seller_rows": len(seller_risk),
        "missing_value_rows": len(missing_values),
        "outlier_rows": len(outliers),
        "leakage_rows": len(leakage_columns),
        "freight_ratio_rows": len(freight_ratio),
        "shipping_limit_days_rows": len(shipping_limit_days),
        "delay_bucket_rows": len(delay_buckets),
        "correlation_rows": len(correlation_matrix),
        "numeric_target_summary_rows": len(numeric_target_summary),
        "rare_category_rows": len(rare_categories),
        "variable_type_summary": variable_type_summary,
    }


if __name__ == "__main__":
    run()
