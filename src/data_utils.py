"""
Ortak veri yardımcı fonksiyonları.

Amaç:
    Farklı pipeline adımlarında tekrar eden işlemleri tek merkezden yönetmek.
"""
import pandas as pd

from src.feature_config import DROP_COLUMNS, TARGET_COLUMN

def prepare_features(df: pd.DataFrame):
    """
    Feature ve target ayrımı yapar.
    Modele verilmemesi gereken kolonları çıkarır.

    Bu fonksiyon preprocessing, model_training ve explainability
    adımlarında aynı feature setinin kullanılmasını sağlar.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    existing_drop_cols = [col for col in DROP_COLUMNS if col in X.columns]
    X = X.drop(columns=existing_drop_cols)

    return X, y
