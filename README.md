# Miuul Data Science Bootcamp 20. Dönem
# DataFu-Group04

### Projedeki Kişiler
- Suha Can
- Aslıhan Gürkan
- Ezgi Özen

### Proje Sunum Tarihi
**19.05.2026 — Salı**

---

## 📋 Proje Kriterleri

### 🎯 Veri Seti Kriterleri
- **Gözlem Sayısı:** Minimum 5.000 satır
- **Değişken Sayısı:** Minimum 10 değişken
- **Kaynak Önerileri:** Kaggle, UCI, TÜİK (özgün veri seçenler ekstra puan kazanır)

### 📊 Puanlama Kriterleri (Toplam: 100 puan + Bonus)

| Kriter | Puan | Açıklama |
|--------|------|----------|
| Proje Veri Seti | 10 | Veri kalitesi & uygunluk |
| EDA & Görselleştirme | 15 | İçgörü çıkarma yeteneği |
| Modelleme | 15 | Doğruluk & metrik seçimi |
| Case Tanımı | 10 | Problem netliği & hedef |
| **Özellik Mühendisliği** | **25** | Feature engineering |
| **Proje Sunumu** | **25** | Anlatım & hikâyeleştirme |
| Farklı Tool Kullanımı (Bonus) | +5 | Streamlit / Flask |

> 💡 Toplam puanın yarısı **Özellik Mühendisliği** ve **Proje Sunumu**'ndan gelmektedir.

### 🎤 Sunum Bilgileri
- **Tarih:** 19 Mayıs 2026 (Salı)
- **Süre:** Maksimum 15 dakika
- **Takım:** Maksimum 4 kişi
- **Teslim:** GitHub repo (zorunlu)

---

*Data-Fu — verinin gücüyle, ustanın disipliniyle.*


# E-Commerce Order Risk Prediction

This project predicts potentially problematic e-commerce orders before fulfillment completion.  
It is designed as a production-like machine learning MVP with preprocessing, model training, FastAPI serving, PostgreSQL logging, Streamlit dashboard, and prediction history tracking.

---

## 1. Business Problem

E-commerce operations may face problematic orders caused by cancellation, unavailability, delivery delay, or customer dissatisfaction.

The goal of this project is to identify risky orders early and help operations teams prioritize manual review.

The model output is not only a prediction. It is converted into:

- risk probability
- risk level
- recommended action
- prediction history log

---

## 2. Dataset

The project uses the Olist Brazilian E-Commerce Dataset.

Main data sources include:

- orders
- order items
- customers
- sellers
- products
- payments
- reviews

The dataset does not contain a direct return label. Therefore, the target variable is engineered.

---

## 3. Target Variable: `problematic_order`

Since there is no real return label, the target variable is created using weak supervision.

A problematic order is identified using historical post-order signals such as:

- low review score
- delivery delay
- canceled or unavailable order status

Important note:

Review score and delivery outcome are used only to create the historical label.  
They are not used as model input features.

This prevents data leakage.

---

## 4. Leakage Control

Leakage-sensitive columns are removed from the model feature set.

Examples:

- review score
- review comments
- delivery delay
- delivered timestamps
- order status
- risk score
- label helper columns

This makes the model closer to a real production scenario where post-order information is not available at prediction time.

---

## 5. Feature Engineering

The project creates operational and historical features such as:

- freight ratio
- estimated delivery days
- payment count
- installment indicator
- multi-item order flag
- multi-seller order flag
- customer historical problem rate
- seller historical problem rate
- category historical problem rate
- purchase month
- purchase hour
- purchase day of week

Historical rate features are designed to avoid leakage by using past behavior instead of future information.

---

## 6. Preprocessing

The preprocessing pipeline includes:

- temporal filtering
- train/test split
- missing value imputation
- robust scaling for numerical variables
- one-hot encoding for categorical variables
- optional SMOTE for imbalanced data
- saving preprocessing artifacts

Saved artifacts:

- `preprocessor.joblib`
- `raw_feature_columns.joblib`
- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `feature_names.csv`

---

## 7. Model Training

Multiple models are benchmarked:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

The project prioritizes Recall and F2-score because missing risky orders is more costly than reviewing some normal orders.

Main metrics:

- Accuracy
- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Threshold tuning is applied to optimize F2-score.

---

## 8. Current Best Model

Current best model:

- Model: CatBoost
- Optimized Recall: 0.5224
- Optimized Precision: 0.2391
- Optimized F1-score: 0.3281
- Optimized F2-score: 0.4223
- Best Threshold: 0.10

Important limitation:

The current model shows overfitting risk.  
Train performance is significantly higher than test performance.

Next model iteration will focus on:

- reducing overfitting
- testing SMOTE alternatives
- tuning CatBoost complexity
- using validation-based threshold tuning
- comparing old vs new results

---

## 9. Explainability

The project includes explainability outputs:

- SHAP summary plot
- feature importance table

Explainability is used for:

- understanding model behavior
- validating business signals
- checking leakage risk
- explaining predictions to stakeholders

---

## 10. Deployment Architecture

The project is implemented as a production-like MVP.

Architecture:

Streamlit UI -> FastAPI Prediction API -> Pipeline.joblib -> 
Optimized Threshold -> PostgreSQL Prediction Logs -> Prediction History Dashboard

Components:
Streamlit: product demo and dashboard
FastAPI: prediction serving layer
PostgreSQL: operational prediction logging
Docker Compose: multi-service deployment
Pipeline.joblib: preprocessor + model inference pipeline

---

## 11. API
FastAPI provides prediction serving.

Main endpoints:
GET /
GET /health
POST /predict
GET /prediction-history

Example Request URL: http://localhost:8000/predict

Example Requests:
{
  "input_data": {
    "customer_zip_code_prefix": 12345,
    "customer_state": "SP",
    "item_count": 1,
    "product_count": 1,
    "total_price": 120.5,
    "total_freight_value": 20.0,
    "payment_type": "credit_card",
    "payment_installments_max": 3,
    "payment_count": 1,
    "total_payment_value": 140.5,
    "customer_historical_problematic": 0,
    "customer_historical_orders": 2,
    "customer_historical_problem_rate": 0.0,
    "purchase_year": 2018,
    "purchase_month": 5,
    "purchase_dayofweek": 2,
    "purchase_hour": 14,
    "is_weekend": 0,
    "is_night_order": 0,
    "order_total_value": 140.5,
    "freight_ratio": 0.17,
    "avg_item_price": 120.5,
    "is_multi_item_order": 0,
    "is_multi_seller_order": 0,
    "is_installment_payment": 1,
    "estimated_delivery_days": 10,
    "approval_delay_hours": 2.0,
    "seller_state_mode": "SP",
    "seller_city_nunique": 1,
    "seller_state_nunique": 1,
    "same_customer_seller_state": 1,
    "main_product_category": "beleza_saude",
    "category_count": 1,
    "is_unknown_category": 0,
    "total_product_weight_g": 500,
    "avg_product_weight_g": 500,
    "max_product_weight_g": 500,
    "total_product_volume_cm3": 3000,
    "avg_product_volume_cm3": 3000,
    "max_product_volume_cm3": 3000,
    "avg_product_photos_qty": 2,
    "avg_product_description_length": 500,
    "seller_historical_problem_count": 1,
    "seller_historical_order_count": 20,
    "seller_historical_problem_rate": 0.05,
    "category_historical_problem_count": 10,
    "category_historical_order_count": 200,
    "category_historical_problem_rate": 0.05,
    "seller_count": 1

  }
}

Example Response:
	
Response body
{
  "risk_probability": 0.020430477289521538,
  "prediction": 0,
  "risk_level": "Low",
  "recommended_action": "Standard Process"
}
Response headers
 content-length: 115 
 content-type: application/json 
 date: Thu,14 May 2026 15:06:08 GMT 
 server: uvicorn 


 ---

## 12. Streamlit Application

The Streamlit app includes:
Executive Dashboard
Order Risk Simulator
Prediction History
Model Performance
Explainability
Business Actions
AI Assistant
Project Story
The Order Risk Simulator sends prediction requests to FastAPI.
FastAPI returns risk

 ---

## 13. PostgreSQL Logging

Prediction requests are stored in the prediction_logs table.
Logged fields:
input data
risk probability
prediction
risk level
recommended action
created timestamp
This enables operational traceability.


 ---

## 14. Docker Usage

Run the project:
docker compose up --build

Services:
FastAPI:   http://localhost:8000
Swagger:   http://localhost:8000/docs
Streamlit: http://localhost:8501
PostgreSQL: localhost:5432

Stop services:
docker compose down

 ---
