"""
Model evaluation modülü.

Amaç:
    Modelin train ve test performansını karşılaştırarak
    overfitting olup olmadığını kontrol etmek.

Cross validation
    
Overfitting:
    Model train setinde çok iyi,
    test setinde belirgin kötü performans gösteriyorsa
    model ezberlemiş olabilir.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class ModelEvaluator:
    """
    Train-test performans karşılaştırması yapan sınıf.
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    def calculate_metrics(self, X, y, dataset_name: str) -> dict:
        """
        Verilen veri seti için metrikleri hesaplar.
        """

        y_pred = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
        else:
            roc_auc = None

        return {
            "model": self.model_name,
            "dataset": dataset_name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "f2_score": fbeta_score(y, y_pred, beta=2, zero_division=0),
            "roc_auc": roc_auc,
        }

    def compare_train_test(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """
        Train ve test metriklerini karşılaştırır.
        """

        train_metrics = self.calculate_metrics(
            X_train,
            y_train,
            dataset_name="train"
        )

        test_metrics = self.calculate_metrics(
            X_test,
            y_test,
            dataset_name="test"
        )

        results = pd.DataFrame([train_metrics, test_metrics])

        return results

    def calculate_overfitting_gap(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Train-test farklarını hesaplar.
        """

        train = results[results["dataset"] == "train"].iloc[0]
        test = results[results["dataset"] == "test"].iloc[0]

        gap_results = {
            "model": self.model_name,
            "accuracy_gap": train["accuracy"] - test["accuracy"],
            "precision_gap": train["precision"] - test["precision"],
            "recall_gap": train["recall"] - test["recall"],
            "f1_gap": train["f1_score"] - test["f1_score"],
            "f2_gap": train["f2_score"] - test["f2_score"],
            "roc_auc_gap": train["roc_auc"] - test["roc_auc"]
            if train["roc_auc"] is not None and test["roc_auc"] is not None
            else None,
        }

        return pd.DataFrame([gap_results])

    def print_test_report(self, X_test, y_test):
        """
        Test seti için detaylı rapor basar.
        """

        y_pred = self.model.predict(X_test)

        print("\n" + "=" * 60)
        print(f"MODEL TEST RAPORU: {self.model_name}")
        print("=" * 60)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))