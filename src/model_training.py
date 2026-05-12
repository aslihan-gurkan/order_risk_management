#6

"""
Model training modülü.

Amaç:
    Preprocessing aşamasından gelen train/test verileriyle
    farklı modelleri eğitmek, karşılaştırmak ve en iyi modeli kaydetmek.

Bu modülde:
    1. Train ve test verisi okunur
    2. Preprocessor yüklenir
    3. Leakage ve ID kolonları çıkarılır
    4. Feature matrisi dönüştürülür
    5. Baseline modeller eğitilir
    6. Metrikler hesaplanır
    7. Threshold tuning yapılır
    8. En iyi model ve metrikler kaydedilir

Neden accuracy ana metrik değil?
    Veri dengesiz olduğu için accuracy yanıltıcı olabilir.
    Bu yüzden PR-AUC, Recall, F1 ve ROC-AUC birlikte değerlendirilir.
"""
import json
from pathlib import Path

from src.data_utils import prepare_features

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from lightgbm import LGBMClassifier

from src.config import (
    PROCESSED_FILES,
    MODEL_FILES,
    RANDOM_STATE,
)
from src.logger import get_logger

logger = get_logger(__name__)


# Model listesi
def get_models() -> dict:
    """
    Karşılaştırılacak baseline modelleri döner.

    Logistic Regression:
        Basit ve açıklanabilir baseline.

    Random Forest:
        Ağaç tabanlı klasik ensemble benchmark.

    LightGBM:
        Tabular veri için güçlü gradient boosting modeli.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            # n_jobs=-1, kaldırıldı çünkü sklearn 1.8’de n_jobs etkisiz uyarısı veriyor.
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    return models


# Metrik hesaplama
def calculate_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Verilen threshold'a göre classification metriklerini hesaplar.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# Threshold tuning
def find_best_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, dict]:
    """
    En iyi threshold değerini bulur.

    Neden 0.5 sabit kullanmıyoruz?
        İş probleminde false negative ve false positive maliyetleri eşit değildir.
        Problematic siparişi kaçırmak çoğu durumda daha maliyetlidir.
    """
    thresholds = np.arange(0.10, 0.91, 0.01) # (0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_metrics = None
    best_score = -1

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_proba, threshold)

        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


# Ana pipeline
def run() -> dict:
    """
    Model training pipeline'ını çalıştırır.
    """
    logger.info("=" * 60)
    logger.info("ADIM 5: Model training başladı")
    logger.info("=" * 60)

    train_path = PROCESSED_FILES["train"]
    test_path = PROCESSED_FILES["test"]

    if not Path(train_path).exists() or not Path(test_path).exists():
        raise FileNotFoundError(
            "Train veya test dosyası bulunamadı. Önce preprocessing.py çalıştırılmalı."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    logger.info(f"Train veri yüklendi: {train_df.shape}")
    logger.info(f"Test veri yüklendi: {test_df.shape}")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    logger.info(f"X_train boyutu: {X_train.shape}")
    logger.info(f"X_test boyutu: {X_test.shape}")
    logger.info(f"Train problematic oranı: {y_train.mean():.2%}")
    logger.info(f"Test problematic oranı: {y_test.mean():.2%}")

    preprocessor_path = MODEL_FILES["preprocessor"]

    if not Path(preprocessor_path).exists():
        raise FileNotFoundError(
            "Preprocessor bulunamadı. Önce preprocessing.py çalıştırılmalı."
        )

    preprocessor = joblib.load(preprocessor_path)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Processed X_train boyutu: {X_train_processed.shape}")
    logger.info(f"Processed X_test boyutu: {X_test_processed.shape}")

    models = get_models()

    results = {}
    best_model_name = None
    best_model = None
    best_score = -1

    for model_name, model in models.items():
        logger.info(f"Model eğitiliyor: {model_name}")

        model.fit(X_train_processed, y_train)

        y_proba = model.predict_proba(X_test_processed)[:, 1]

        default_metrics = calculate_metrics(
            y_true=y_test,
            y_proba=y_proba,
            threshold=0.5,
        )

        best_threshold, tuned_metrics = find_best_threshold(
            y_true=y_test,
            y_proba=y_proba,
            metric="f1",
        )

        results[model_name] = {
            "default_threshold_metrics": default_metrics,
            "best_threshold": best_threshold,
            "tuned_metrics": tuned_metrics,
        }

        logger.info(
            f"{model_name} sonuçları | "
            f"ROC-AUC: {tuned_metrics['roc_auc']:.4f} | "
            f"PR-AUC: {tuned_metrics['pr_auc']:.4f} | "
            f"Precision: {tuned_metrics['precision']:.4f} | "
            f"Recall: {tuned_metrics['recall']:.4f} | "
            f"F1: {tuned_metrics['f1']:.4f} | "
            f"Threshold: {best_threshold:.2f}"
        )

        # Ana seçim metriği PR-AUC
        model_score = tuned_metrics["pr_auc"]

        if model_score > best_score:
            best_score = model_score
            best_model_name = model_name
            best_model = model

    output = {
        "best_model_name": best_model_name,
        "selection_metric": "pr_auc",
        "best_score": float(best_score),
        "results": results,
    }

    joblib.dump(best_model, MODEL_FILES["best_model"])

    with open(MODEL_FILES["metrics"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    logger.info(f"En iyi model: {best_model_name}")
    logger.info(f"En iyi PR-AUC: {best_score:.4f}")
    logger.info(f"Model kaydedildi: {MODEL_FILES['best_model']}")
    logger.info(f"Metrikler kaydedildi: {MODEL_FILES['metrics']}")

    logger.info("=" * 60)
    logger.info("ADIM 5 tamamlandı")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    run()