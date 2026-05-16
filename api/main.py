from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from api.services.prediction_service import predict_order_risk

from api.database import Base, engine, get_db
from api.models import PredictionLog
from api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionLogResponse
)

app = FastAPI(
    title="Ecommerce Order Risk API",
    description="FastAPI service for problematic order risk prediction",
    version="1.0.0",
)


Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):

    prediction_result = predict_order_risk(request.input_data)

    log = PredictionLog(
        input_data=request.input_data,
        risk_probability=prediction_result["risk_probability"],
        prediction=prediction_result["prediction"],
        risk_level=prediction_result["risk_level"],
        recommended_action=prediction_result["recommended_action"],
    )

    db.add(log)
    db.commit()

    return PredictionResponse(**prediction_result)

@app.get("/prediction-history", response_model=list[PredictionLogResponse])
def get_prediction_history(db: Session = Depends(get_db)):
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(50)
        .all()
    )

    return logs