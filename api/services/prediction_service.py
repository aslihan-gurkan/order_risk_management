from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

from src.config import MODEL_FILES, MODELS_PATH


pipeline = joblib.load(MODEL_FILES["pipeline"])

THRESHOLD_PATH = Path(MODELS_PATH) / "decision_threshold.joblib"
RAW_FEATURE_COLUMNS_PATH = Path(MODELS_PATH) / "raw_feature_columns.joblib"

decision_threshold = joblib.load(THRESHOLD_PATH)
raw_feature_columns = joblib.load(RAW_FEATURE_COLUMNS_PATH)


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"


def get_recommended_action(risk_level: str) -> str:
    if risk_level == "High":
        return "Manual Review"
    elif risk_level == "Medium":
        return "Monitor"
    return "Standard Process"


def validate_input(input_data: dict) -> None:
    missing_cols = [
        col for col in raw_feature_columns
        if col not in input_data
    ]

    extra_cols = [
        col for col in input_data.keys()
        if col not in raw_feature_columns
    ]

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required model features",
                "missing_columns": missing_cols,
                "extra_columns": extra_cols,
            },
        )


def predict_order_risk(input_data: dict) -> dict:
    validate_input(input_data)

    input_df = pd.DataFrame([input_data])
    input_df = input_df[raw_feature_columns]

    probability = pipeline.predict_proba(input_df)[0][1]
    prediction = int(probability >= decision_threshold)

    risk_level = get_risk_level(probability)
    recommended_action = get_recommended_action(risk_level)

    return {
        "risk_probability": float(probability),
        "prediction": prediction,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
    }