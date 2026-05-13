"""
Model explainability modülü.

Amaç:
    Eğitilen en iyi modelin kararlarını yorumlamak.

Bu modülde:
    1. Best model yüklenir
    2. Preprocessor yüklenir
    3. Test verisi hazırlanır
    4. Feature importance çıkarılır
    5. SHAP analizi yapılır
    6. Sonuçlar data/outputs altına kaydedilir

Neden önemli?
    Model sadece tahmin üretmemelidir.
    Operasyon ekipleri riskin neden oluştuğunu da anlamalıdır.
"""
from pathlib import Path

from src.data_utils import prepare_features

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import PROCESSED_FILES, MODEL_FILES, OUTPUTS_PATH
from src.logger import get_logger

logger = get_logger(__name__)

def get_feature_names(preprocessor) -> list[str]:
    """
    Preprocessor sonrası oluşan feature isimlerini üretir.

    Neden gerekli?
        Modelin gördüğü veri artık raw kolonlar değildir.
        OneHotEncoder sonrası kolon sayısı artar.
        Feature importance ve SHAP için bu isimleri yeniden üretmemiz gerekir.
    """
    feature_names = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue

        if transformer_name == "numerical":
            feature_names.extend(columns)

        elif transformer_name == "low_cardinality":
            onehot = transformer.named_steps["onehot"]
            encoded_names = onehot.get_feature_names_out(columns)
            feature_names.extend(encoded_names.tolist())

        elif transformer_name == "high_cardinality":
            feature_names.extend(columns)

    return feature_names


def save_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Modelin feature importance değerlerini kaydeder.

    Random Forest ve LightGBM gibi ağaç tabanlı modellerde
    feature_importances_ kullanılır.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model feature_importances_ desteklemiyor.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    output_path = Path(OUTPUTS_PATH) / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    logger.info(f"Feature importance kaydedildi: {output_path}")

    return importance_df


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    """
    En önemli 20 feature için bar chart kaydeder.
    """
    if importance_df.empty:
        return

    top_features = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    output_path = Path(OUTPUTS_PATH) / "feature_importance_top20.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Feature importance grafiği kaydedildi: {output_path}")


def calculate_shap_values(model, X_processed, feature_names: list[str]) -> pd.DataFrame:
    """
    SHAP değerlerini hesaplar.

    Performans için tüm test seti yerine örneklem kullanılır.
    SHAP özellikle ağaç tabanlı modellerde karar açıklamak için güçlüdür.
    """
    sample_size = min(1000, X_processed.shape[0])

    X_sample = X_processed[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_values_class_1 = shap_values[:, :, 1]
    else:
        shap_values_class_1 = shap_values

    mean_abs_shap = np.abs(shap_values_class_1).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    shap_df = shap_df.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    output_path = Path(OUTPUTS_PATH) / "shap_feature_importance.csv"
    shap_df.to_csv(output_path, index=False)

    logger.info(f"SHAP feature importance kaydedildi: {output_path}")

    return shap_df


def save_shap_plot(shap_df: pd.DataFrame) -> None:
    """
    SHAP önem sıralamasını grafik olarak kaydeder.
    """
    top_features = shap_df.head(20).sort_values("mean_abs_shap")

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"])
    plt.title("Top 20 SHAP Feature Importance")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()

    output_path = Path(OUTPUTS_PATH) / "shap_feature_importance_top20.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"SHAP grafiği kaydedildi: {output_path}")


def run() -> dict:
    """
    Explainability pipeline'ını çalıştırır.
    """
    logger.info("=" * 60)
    logger.info("ADIM 6: Model explainability başladı")
    logger.info("=" * 60)

    Path(OUTPUTS_PATH).mkdir(parents=True, exist_ok=True)

    test_path = PROCESSED_FILES["test"]

    if not Path(test_path).exists():
        raise FileNotFoundError("Test dosyası bulunamadı. Önce preprocessing çalıştırılmalı.")

    if not Path(MODEL_FILES["best_model"]).exists():
        raise FileNotFoundError("Best model bulunamadı. Önce model_training çalıştırılmalı.")

    if not Path(MODEL_FILES["preprocessor"]).exists():
        raise FileNotFoundError("Preprocessor bulunamadı. Önce preprocessing çalıştırılmalı.")

    test_df = pd.read_parquet(test_path)

    model = joblib.load(MODEL_FILES["best_model"])
    preprocessor = joblib.load(MODEL_FILES["preprocessor"])

    X_test, y_test = prepare_features(test_df)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    logger.info(f"Test veri boyutu: {test_df.shape}")
    logger.info(f"Processed test boyutu: {X_test_processed.shape}")
    logger.info(f"Feature sayısı: {len(feature_names)}")

    importance_df = save_feature_importance(model, feature_names)
    save_feature_importance_plot(importance_df)

    shap_df = calculate_shap_values(
        model=model,
        X_processed=X_test_processed,
        feature_names=feature_names,
    )
    save_shap_plot(shap_df)

    logger.info("En önemli 10 SHAP feature:")
    for _, row in shap_df.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['mean_abs_shap']:.6f}")

    logger.info("=" * 60)
    logger.info("ADIM 6 tamamlandı")
    logger.info("=" * 60)

    return {
        "feature_importance_path": str(Path(OUTPUTS_PATH) / "feature_importance.csv"),
        "shap_importance_path": str(Path(OUTPUTS_PATH) / "shap_feature_importance.csv"),
    }


if __name__ == "__main__":
    run()