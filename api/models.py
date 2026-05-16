from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func

from api.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    input_data = Column(JSON, nullable=False)

    risk_probability = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)
    risk_level = Column(String(50), nullable=False)
    recommended_action = Column(String(100), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())