"""
Model training modülü.

Amaç:
    Preprocessing sonrası oluşturulan model input dosyalarını kullanarak
    farklı modelleri benchmark etmek ve en iyi modeli kaydetmek.

Bu modülde:
    1. X_train, X_test, y_train, y_test okunur
    2. Birden fazla model eğitilir
    3. Recall, Precision, F1, F2, ROC-AUC ve PR-AUC metrikleri hesaplanır
    4. Threshold=0.50 ile temel sonuçlar alınır
    5. F2-score odaklı threshold optimizasyonu yapılır
    6. En iyi model kaydedilir
    7. Model karşılaştırma çıktıları CSV/JSON olarak yazılır

Kritik prensip:
    Bu problemde amaç sadece yüksek accuracy almak değildir.
    Problematic order sınıfını yakalamak önemli olduğu için Recall ve F2-score
    daha öncelikli değerlendirilir.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    MODEL_FILES,
    OUTPUTS_PATH,
    RANDOM_STATE,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Genel ayarlar
# ─────────────────────────────────────────────────────────────────────────────
MODEL_INPUT_PATH = Path("data/processed/model_input")
MODEL_OUTPUT_PATH = Path(OUTPUTS_PATH) / "models"
METRICS_OUTPUT_PATH = Path(OUTPUTS_PATH) / "metrics"

BEST_MODEL_PATH = Path(MODEL_FILES["best_model"])
MODEL_METRICS_PATH = Path(MODEL_FILES["metrics"])

TARGET_COL = "problematic_order"
POSITIVE_CLASS = 1

# Business problem gereği recall ve F2 daha öncelikli.
SELECTION_METRIC = "f2_optimized"


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_dirs() -> None:
    """Model çıktı klasörlerini oluşturur."""
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_model_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Preprocessing çıktısı olan train/test numpy dosyalarını okur."""
    required_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "feature_names.csv",
    ]

    missing_files = [
        file for file in required_files
        if not (MODEL_INPUT_PATH / file).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Model input dosyaları eksik: {missing_files}. "
            "Önce preprocessing çalıştırılmalı: python -m src.preprocessing"
        )

    X_train = np.load(MODEL_INPUT_PATH / "X_train.npy")
    X_test = np.load(MODEL_INPUT_PATH / "X_test.npy")
    y_train = np.load(MODEL_INPUT_PATH / "y_train.npy")
    y_test = np.load(MODEL_INPUT_PATH / "y_test.npy")

    feature_names = pd.read_csv(MODEL_INPUT_PATH / "feature_names.csv")["feature_name"].tolist()

    logger.info(f"X_train yüklendi: {X_train.shape}")
    logger.info(f"X_test yüklendi: {X_test.shape}")
    logger.info(f"y_train dağılımı: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    logger.info(f"y_test dağılımı: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
    logger.info(f"Feature sayısı: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, feature_names


def get_models() -> Dict[str, object]:
    """
    Benchmark edilecek modelleri döndürür.

    Not:
        XGBoost, LightGBM, CatBoost opsiyoneldir.
        Kurulu değilse pipeline bozulmaz; sadece ilgili model atlanır.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost kurulu değil, XGBoost benchmark dışı bırakıldı.")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("lightgbm kurulu değil, LightGBM benchmark dışı bırakıldı.")

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="F1",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
    except ImportError:
        logger.warning("catboost kurulu değil, CatBoost benchmark dışı bırakıldı.")

    return models


def get_positive_probabilities(model, X_test: np.ndarray) -> np.ndarray:
    """Modelden positive class probability değerlerini alır."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1 / (1 + np.exp(-scores))

    # Son çare: predict çıktısını probability gibi kullanır.
    return model.predict(X_test)


def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Belirli threshold için classification metriklerini hesaplar."""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def optimize_threshold_for_f2(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Dict[str, float]:
    """F2-score'u maksimize eden threshold değerini bulur."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []
    for threshold in thresholds:
        metric_row = calculate_metrics(y_true, y_proba, threshold=float(threshold))
        rows.append(metric_row)

    result = pd.DataFrame(rows)
    best_row = result.sort_values("f2", ascending=False).iloc[0].to_dict()

    return best_row


def train_and_evaluate_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, Dict]]:
    """Tüm modelleri eğitir ve test set üzerinde değerlendirir."""
    results = []
    trained_models = {}
    model_details = {}

    for model_name, model in models.items():
        logger.info("-" * 60)
        logger.info(f"Model eğitimi başladı: {model_name}")

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        y_proba = get_positive_probabilities(model, X_test)

        default_metrics = calculate_metrics(y_test, y_proba, threshold=0.50)
        optimized_metrics = optimize_threshold_for_f2(y_test, y_proba)

        result_row = {
            "model_name": model_name,
            "accuracy_default": default_metrics["accuracy"],
            "precision_default": default_metrics["precision"],
            "recall_default": default_metrics["recall"],
            "f1_default": default_metrics["f1"],
            "f2_default": default_metrics["f2"],
            "roc_auc": default_metrics["roc_auc"],
            "pr_auc": default_metrics["pr_auc"],
            "best_threshold": optimized_metrics["threshold"],
            "precision_optimized": optimized_metrics["precision"],
            "recall_optimized": optimized_metrics["recall"],
            "f1_optimized": optimized_metrics["f1"],
            "f2_optimized": optimized_metrics["f2"],
            "tp_optimized": optimized_metrics["tp"],
            "fp_optimized": optimized_metrics["fp"],
            "fn_optimized": optimized_metrics["fn"],
            "tn_optimized": optimized_metrics["tn"],
        }

        results.append(result_row)

        model_details[model_name] = {
            "default_metrics": default_metrics,
            "optimized_metrics": optimized_metrics,
        }

        logger.info(
            f"{model_name} | "
            f"Recall={optimized_metrics['recall']:.4f}, "
            f"Precision={optimized_metrics['precision']:.4f}, "
            f"F1={optimized_metrics['f1']:.4f}, "
            f"F2={optimized_metrics['f2']:.4f}, "
            f"PR-AUC={default_metrics['pr_auc']:.4f}, "
            f"Best threshold={optimized_metrics['threshold']:.2f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(SELECTION_METRIC, ascending=False)

    return results_df, trained_models, model_details


def save_results(
    results_df: pd.DataFrame,
    trained_models: Dict[str, object],
    model_details: Dict[str, Dict],
    feature_names: List[str],
) -> None:
    """Model sonuçlarını ve en iyi modeli kaydeder."""
    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]

    best_model_metrics = results_df.iloc[0].to_dict()

    results_path = METRICS_OUTPUT_PATH / "model_benchmark_results.csv"
    details_path = METRICS_OUTPUT_PATH / "model_details.json"
    feature_path = METRICS_OUTPUT_PATH / "model_feature_names.csv"

    results_df.to_csv(results_path, index=False)

    serializable_details = {
        model_name: {
            section: {
                key: float(value) if isinstance(value, (np.float32, np.float64, float)) else int(value) if isinstance(value, (np.integer, int)) else value
                for key, value in metrics.items()
            }
            for section, metrics in detail.items()
        }
        for model_name, detail in model_details.items()
    }

    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(serializable_details, file, indent=4, ensure_ascii=False)

    pd.Series(feature_names, name="feature_name").to_csv(feature_path, index=False)

    joblib.dump(best_model, BEST_MODEL_PATH)

    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(best_model_metrics, file, indent=4, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"En iyi model: {best_model_name}")
    logger.info(f"En iyi model kaydedildi: {BEST_MODEL_PATH}")
    logger.info(f"Benchmark sonuçları kaydedildi: {results_path}")
    logger.info(f"Model metrikleri kaydedildi: {MODEL_METRICS_PATH}")
    logger.info("=" * 60)


def run() -> pd.DataFrame:
    """Model training pipeline'ını çalıştırır."""
    logger.info("=" * 60)
    logger.info("ADIM 6: Model training başladı")
    logger.info("=" * 60)

    ensure_output_dirs()

    X_train, X_test, y_train, y_test, feature_names = load_model_inputs()
    models = get_models()

    logger.info(f"Benchmark edilecek model sayısı: {len(models)}")
    logger.info(f"Modeller: {list(models.keys())}")

    results_df, trained_models, model_details = train_and_evaluate_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    logger.info("Model benchmark sonuçları:")
    logger.info(results_df.to_string(index=False))

    save_results(
        results_df=results_df,
        trained_models=trained_models,
        model_details=model_details,
        feature_names=feature_names,
    )

    logger.info("=" * 60)
    logger.info("ADIM 6 tamamlandı")
    logger.info("=" * 60)

    return results_df


if __name__ == "__main__":
    run()
