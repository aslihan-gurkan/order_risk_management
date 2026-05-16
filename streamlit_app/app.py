from pathlib import Path
import shap
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="E-Commerce Order Risk Prediction",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

METRICS_PATH = BASE_DIR / "data" / "outputs" / "metrics" / "model_benchmark_results.csv"
CV_PATH = BASE_DIR / "data" / "outputs" / "metrics" / "cross_validation_results.csv"
SHAP_PATH = BASE_DIR / "data" / "outputs" / "explainability" / "shap_summary.png"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "outputs" / "explainability" / "feature_importance.csv"
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
FEATURE_NAMES_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "model_input" /
    "feature_names.csv"
)
#PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
FEATURED_DATA_PATH = BASE_DIR / "data" / "processed" / "featured.parquet"
API_URL = os.getenv("API_URL", "http://api:8000/predict")

# ============================================================
# Helper Functions
# ============================================================

def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


def get_best_model(metrics_df):
    if metrics_df.empty:
        return None
    return metrics_df.iloc[0]

@st.cache_data
def load_feature_columns():
    if not FEATURED_DATA_PATH.exists():
        return []

    df = pd.read_parquet(FEATURED_DATA_PATH)

    drop_cols = [
        "problematic_order",
        "order_id",
        "customer_id",
        "customer_unique_id",
        "main_seller_id",
        "product_id",
        "review_score",
        "risk_score",
        "order_status",
    ]

    feature_cols = [col for col in df.columns if col not in drop_cols]

    return feature_cols

def call_prediction_api(input_data: dict):
    payload = {"input_data": input_data}

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()

        st.error(f"API Error: {response.status_code}")
        st.json(response.json())
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return None

@st.cache_resource
def load_shap_artifacts():
    pipeline = joblib.load(BASE_DIR / "models" / "pipeline.joblib")
    raw_feature_columns = joblib.load(BASE_DIR / "models" / "raw_feature_columns.joblib")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = list(preprocessor.get_feature_names_out())

    # XGBoost base_score bug fix
    if hasattr(model, "get_booster"):
        booster = model.get_booster()

        config = json.loads(booster.save_config())
        base_score = config["learner"]["learner_model_param"].get("base_score")

        if isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
            config["learner"]["learner_model_param"]["base_score"] = base_score.strip("[]")
            booster.load_config(json.dumps(config))

        explainer = shap.TreeExplainer(booster)
        shap_model = booster
    else:
        explainer = shap.TreeExplainer(model)
        shap_model = model

    return pipeline, preprocessor, shap_model, explainer, feature_names, raw_feature_columns


def call_prediction_history_api():
    try:
        response = requests.get(HISTORY_API_URL, timeout=10)

        if response.status_code == 200:
            return response.json()

        st.error(f"API Error: {response.status_code}")
        st.json(response.json())
        return []

    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return []

# w/out api call - 
def predict_real_risk(input_df):
    model, preprocessor = load_model_artifacts()

    if model is None or preprocessor is None:
        return None

    X_processed = preprocessor.transform(input_df)

    probability = model.predict_proba(X_processed)[0][1]
    prediction = model.predict(X_processed)[0]

    return probability, prediction


def get_risk_level(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.55:
        return "Medium"
    elif probability < 0.75:
        return "High"
    return "Critical"


def get_recommended_action(risk_level):
    actions = {
        "Low": "Normal flow. No extra action required.",
        "Medium": "Monitor order and check delivery timeline.",
        "High": "Seller confirmation + delivery follow-up recommended.",
        "Critical": "Manual review required before order proceeds.",
    }
    return actions.get(risk_level, "No action available.")

def prettify_feature_name(feature_name):
    feature_name = (
        feature_name
        .replace("num__", "")
        .replace("cat__", "")
        .replace("remainder__", "")
    )
    feature_dictionary = {
        "purchase_month": "Satın Alma Ayı",
        "purchase_hour": "Satın Alma Saati",
        "purchase_dayofweek": "Satın Alma Günü",
        "estimated_delivery_days": "Tahmini Teslim Süresi",
        "seller_historical_problem_count": "Satıcının Geçmiş Problemli Sipariş Sayısı",
        "seller_historical_problem_rate": "Satıcının Geçmiş Problem Oranı",
        "category_historical_problem_rate": "Kategori Geçmiş Problem Oranı",
        "payment_installments_max": "Maksimum Taksit Sayısı",
        "avg_product_photos_qty": "Ortalama Ürün Fotoğraf Sayısı",
        "freight_ratio": "Kargo / Sipariş Tutarı Oranı",
        "avg_item_price": "Ortalama Ürün Fiyatı",
        "customer_state": "Müşteri Bölgesi",
        "seller_state_mode": "Satıcı Bölgesi",
        "seller_state_nunique": "Satıcı Bölge Çeşitliliği",
        "seller_city_nunique": "Satıcı Şehir Çeşitliliği",
        "same_customer_seller_state": "Müşteri ve Satıcı Aynı Bölgede",
        "is_multi_seller_order": "Çoklu Satıcılı Sipariş",
        "is_multi_item_order": "Çoklu Ürünlü Sipariş",
        "payment_count": "Ödeme Sayısı",
        "order_total_value": "Sipariş Toplam Tutarı",
    }

    clean_name = feature_name

    # One-hot encoded kolonları anlamlı hale getirir
    for prefix in feature_dictionary.keys():
        if clean_name == prefix or clean_name.startswith(prefix + "_"):
            suffix = clean_name.replace(prefix + "_", "")

            if suffix in ["0", "1"]:
                return f"{feature_dictionary[prefix]} = {suffix}"

            if len(suffix) > 0 and suffix != clean_name:
                return f"{feature_dictionary[prefix]}: {suffix}"

            return feature_dictionary[prefix]

    return feature_name

def render_kpi_cards(metrics_df):
    best_model = get_best_model(metrics_df)

    col1, col2, col3, col4 = st.columns(4)

    if best_model is not None:
        col1.metric("Best Model", best_model.get("model_name", "-"))
        col2.metric("Recall", round(best_model.get("recall_optimized", 0), 4))
        col3.metric("F1-score", round(best_model.get("f1_optimized", 0), 4))
        col4.metric("F2-score", round(best_model.get("f2_optimized", 0), 4))
    else:
        col1.metric("Best Model", "-")
        col2.metric("Recall", "-")
        col3.metric("F1-score", "-")
        col4.metric("F2-score", "-")

    st.info(
        "Model, operasyon ekibinin riskli siparişleri önceliklendirmesine yardımcı olmak için tasarlanmıştır. "
        "Leakage temizliği sonrası skorlar daha gerçekçi hale gelmiştir."
    )


def ai_assistant_answer(question):
    q = question.lower()

    if "f2" in q:
        return """
        F2-score, recall'a daha fazla ağırlık veren bir metriktir.
        Bu projede riskli siparişleri kaçırmamak önemli olduğu için F2-score takip edilmiştir.
        """

    if "recall" in q:
        return """
        Recall, gerçekten riskli olan siparişlerin ne kadarını yakaladığımızı gösterir.
        Operasyonel risk probleminde recall önemlidir çünkü riskli siparişi kaçırmak maliyetli olabilir.
        """

    if "leakage" in q or "performans" in q or "performance" in q:
        return """
        Leakage temizlendikten sonra model artık review score, gerçek teslimat gecikmesi
        veya sipariş tamamlandıktan sonra oluşan bilgileri kullanmaz.
        Bu yüzden performans düşebilir; fakat yeni skorlar production senaryosuna daha yakındır.
        """

    if "production" in q or "ready" in q:
        return """
        Bu model MVP seviyesinde production demosuna yakındır; fakat gerçek production için
        model monitoring, drift takibi, API logging, threshold tuning ve canlı veri validasyonu gerekir.
        """

    if "risk" in q or "risky" in q or "neden" in q:
        return """
        Bir sipariş genellikle yüksek freight_ratio, uzun estimated_delivery_days,
        yüksek seller/category historical problem rate veya multi-seller yapısı nedeniyle riskli görünebilir.
        """

    if "operation" in q or "aksiyon" in q or "action" in q:
        return """
        Operasyon ekibi risk seviyesine göre aksiyon alabilir:
        Low: normal akış,
        Medium: takip,
        High: seller confirmation + delivery follow-up,
        Critical: manual review.
        """

    return """
    Bu assistant şu an kontrollü rule-based çalışıyor.
    Yani sadece proje mantığı, metrikler, leakage, risk seviyesi ve operasyon aksiyonları hakkında cevap verir.
    Sonraki iterasyonda local LLM veya RAG yapısına bağlanabilir.
    """

@st.cache_data
def load_featured_data():
    if FEATURED_DATA_PATH.exists():
        return pd.read_parquet(FEATURED_DATA_PATH)
    return pd.DataFrame()


def build_data_context(df):
    if df.empty:
        return "Dataset could not be loaded."

    total_orders = len(df)
    risk_rate = df["problematic_order"].mean()

    top_categories = (
        df.groupby("main_product_category")["problematic_order"]
        .agg(["count", "mean"])
        .reset_index()
        .query("count >= 50")
        .sort_values("mean", ascending=False)
        .head(5)
    )

    top_states = (
        df.groupby("customer_state")["problematic_order"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(5)
    )

    # YENİ — seller verisi ekle
    top_sellers = (
        df.groupby("main_seller_id")["problematic_order"]
        .agg(["count", "mean"])
        .reset_index()
        .query("count >= 20")
        .sort_values("mean", ascending=False)
        .head(5)
    )

    context = f"""
Dataset summary:
- Total orders: {total_orders}
- Overall problematic order rate: {risk_rate:.2%}

Top risky categories:
{top_categories.to_string(index=False)}

Top risky customer states:
{top_states.to_string(index=False)}

Top risky sellers (min 20 orders):
{top_sellers.to_string(index=False)}
"""
    return context

def local_dataframe_answer(question, df):
    q = question.lower()

    if df.empty:
        return "Dataset yüklenemedi."

    if ( "category" in q or "categories" in q or "kategori" in q or "product" in q):
        result = (
            df.groupby("main_product_category")["problematic_order"]
            .agg(
                total_orders="count",
                risk_rate="mean"
            )
            .reset_index()
            .query("main_product_category not in ['unknown']")
            .query("total_orders >= 50")
            .sort_values("risk_rate", ascending=False)
            .head(10)
        )
        result["risk_percentage"] = (
            result["risk_rate"] * 100
        ).round(2)

        result = result.drop(columns=["risk_rate"])

        result = result.rename(
            columns={
                "main_product_category": "product_category"
            }
        )
        result["product_category"] = (
            result["product_category"]
            .str.replace("_", " ", regex=False)
            .str.title()
        )
        return result

    if (
        "state" in q
        or "states" in q
        or "region" in q
        or "regions" in q
        or "eyalet" in q
        or "bölge" in q
    ):
        result = (
            df.groupby("customer_state")["problematic_order"]
            .agg(
                total_orders="count",
                risk_rate="mean"
            )
            .reset_index()
            .query("total_orders >= 50")
            .sort_values("risk_rate", ascending=False)
            .head(10)
        )

        result["risk_percentage"] = (result["risk_rate"] * 100).round(2)
        result = result.drop(columns=["risk_rate"])

        result = result.rename(
            columns={
                "customer_state": "customer_region"
            }
        )

        return result

    if (
        "seller" in q
        or "sellers" in q
        or "satıcı" in q
    ):
        result = (
            df.groupby("main_seller_id")["problematic_order"]
            .agg(
                total_orders="count",
                risk_rate="mean"
            )
            .reset_index()
            .query("total_orders >= 20")
            .sort_values("risk_rate", ascending=False)
            .head(10)
        )

        result["risk_percentage"] = (result["risk_rate"] * 100).round(2)
        result = result.drop(columns=["risk_rate"])

        result = result.rename(
            columns={
                "main_seller_id": "seller_id"
            }
        )

        return result

    if (
        "freight" in q
        or "shipping" in q
        or "cargo" in q
        or "kargo" in q
    ):
        threshold = df["freight_ratio"].quantile(0.75)

        high = df[df["freight_ratio"] > threshold]
        low = df[df["freight_ratio"] <= threshold]

        result = pd.DataFrame(
            {
                "segment": [
                    "High freight ratio orders",
                    "Other orders"
                ],
                "total_orders": [
                    len(high),
                    len(low)
                ],
                "risk_percentage": [
                    round(high["problematic_order"].mean() * 100, 2),
                    round(low["problematic_order"].mean() * 100, 2),
                ],
            }
        )

        return result

    return """
    Bu soru için local dataframe assistant şu an kategori, state, seller ve freight analizlerini destekliyor.
    OpenAI veya Gemini moduna geçersen daha serbest açıklama alabilirsin.
    """


def openai_answer(question, df):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    context = build_data_context(df)

    prompt = f"""
        You are an Operational AI Analytics Assistant for an e-commerce order risk prediction platform.

        Your role is to help operations teams, business stakeholders, and analysts understand:
        - risky orders
        - operational risk patterns
        - seller and category behavior
        - model decisions
        - explainability outputs
        - business impact of model predictions

        IMPORTANT RULES:
        1. Answer ONLY using the provided dataset context and project information.
        2. Never hallucinate or invent metrics, counts, percentages, or business findings.
        3. If the information is not available in the provided context, explicitly say so.
        4. Do not claim access to real-time systems or external databases.
        5. Keep explanations operationally meaningful and business-oriented.
        6. Explain technical concepts in a way that non-technical stakeholders can understand.
        7. When discussing model performance:
        - emphasize Recall and F2-score importance
        - explain trade-offs between Precision and Recall
        - mention leakage prevention when relevant
        8. When discussing risk:
        - focus on operational risk signals
        - avoid deterministic language
        - explain risk as probability, not certainty
        9. When relevant, provide actionable operational recommendations.

        PROJECT CONTEXT:
        - Project type: E-commerce operational risk prediction
        - Main objective: identify potentially problematic orders before fulfillment completion
        - Target variable: problematic_order
        - Dataset: Olist Brazilian E-Commerce Dataset
        - Modeling approach: supervised binary classification
        - Explainability methods: SHAP + feature importance analysis
        - Deployment stack: Streamlit, FastAPI, PostgreSQL, Docker
        - Key business goal: prioritize operational review and reduce problematic order impact

        IMPORTANT MODELING NOTES:
        - Review score and real delivery outcomes are used only during historical label engineering.
        - Leakage-sensitive variables are removed from prediction features.
        - Model evaluation prioritizes Recall and F2-score because missing risky orders is operationally costly.
        - Explainability outputs are also used as a leakage validation layer.

        DATASET CONTEXT:
        {context}

        USER QUESTION:
        {question}
        """

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text


def gemini_answer(question, df):
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    context = build_data_context(df)

    prompt = f"""
You are an AI assistant for an e-commerce order risk prediction project.

Answer based only on the project context and dataset summary below.
Do not invent numbers that are not given.

Dataset context:
{context}

User question:
{question}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📦 Order Risk App")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Order Risk Simulator",
        "Model Performance",
        "Explainability",
        "Business Actions",
        "AI Assistant",
        "Project Story",
        "Prediction History"
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Product")
st.sidebar.caption("E-Commerce Order Risk Prediction")


# ============================================================
# Data
# ============================================================

metrics_df = load_metrics()


# ============================================================
# Pages
# ============================================================

if page == "Dashboard":
    st.title("📦 Executive Dashboard")
    st.caption("Operational overview for e-commerce problematic order risk prediction")

    df = load_featured_data()

    if df.empty:
        st.error("Featured dataset bulunamadı.")
    else:
        total_orders = len(df)
        risky_order_rate = df["problematic_order"].mean()
        avg_freight_ratio = df["freight_ratio"].mean()
        avg_estimated_delivery_days = df["estimated_delivery_days"].mean()

        top_risky_state = (
            df.groupby("customer_state")["problematic_order"]
            .agg(total_orders="count", risk_rate="mean")
            .reset_index()
            .query("total_orders >= 50")
            .sort_values("risk_rate", ascending=False)
            .iloc[0]
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Orders", f"{total_orders:,}")
        col2.metric("Risky Order Rate", f"{risky_order_rate:.2%}")
        col3.metric("Avg Freight Ratio", f"{avg_freight_ratio:.2%}")
        col4.metric("Avg Delivery Days", f"{avg_estimated_delivery_days:.1f}")
        col5.metric("Top Risky State", top_risky_state["customer_state"])

        st.info(
            """
            Bu dashboard, operasyon ekibinin sipariş riskini hızlıca izlemesi,
            riskin yoğunlaştığı kategori/bölge alanlarını görmesi ve model çıktısını
            iş aksiyonlarına çevirmesi için tasarlanmıştır.
            """
        )

        st.markdown("### Risk Overview")

        col1, col2 = st.columns(2)

        with col1:
            risk_distribution = (
                df["problematic_order"]
                .value_counts()
                .rename(index={0: "Normal Orders", 1: "Problematic Orders"})
                .reset_index()
            )

            risk_distribution.columns = ["order_type", "order_count"]

            st.bar_chart(
                risk_distribution,
                x="order_type",
                y="order_count",
                use_container_width=True
            )

        with col2:
            top_categories = (
                df.groupby("main_product_category")["problematic_order"]
                .agg(total_orders="count", risk_rate="mean")
                .reset_index()
                .query("main_product_category != 'unknown'")
                .query("total_orders >= 50")
                .sort_values("risk_rate", ascending=False)
                .head(10)
            )

            top_categories["risk_percentage"] = (
                top_categories["risk_rate"] * 100
            ).round(2)

            top_categories["main_product_category"] = (
                top_categories["main_product_category"]
                .str.replace("_", " ", regex=False)
                .str.title()
            )

            st.bar_chart(
                top_categories,
                x="main_product_category",
                y="risk_percentage",
                use_container_width=True
            )

        st.markdown("### Monthly Risk Trend")

        monthly_risk = (
            df.groupby("purchase_month")["problematic_order"]
            .agg(total_orders="count", risk_rate="mean")
            .reset_index()
            .sort_values("purchase_month")
        )

        monthly_risk["risk_percentage"] = (
            monthly_risk["risk_rate"] * 100
        ).round(2)

        st.line_chart(
            monthly_risk,
            x="purchase_month",
            y="risk_percentage",
            use_container_width=True
        )

        st.markdown("### Operational Insight")

        st.success(
            """
            Son tahminler operasyonel olarak PostgreSQL üzerinde kayıt altına alınmaktadır.

            Sistem:
            - risk skorunu hesaplar
            - aksiyon önerisi üretir
            - tahmin geçmişini saklar
            - operasyon ekibine karar desteği sağlar
            """
        )



elif page == "Order Risk Simulator":
    st.title("🧪 Order Risk Simulator")
    st.caption("Manuel input ile sipariş risk seviyesini simüle eder.")

    col1, col2 = st.columns(2)

    with col1:
        order_total_value = st.number_input(
            "Order Total Value",
            min_value=0.0,
            value=250.0,
            step=50.0,
        )

        freight_ratio = st.slider(
            "Freight Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
        )

        estimated_delivery_days = st.number_input(
            "Estimated Delivery Days",
            min_value=1,
            value=8,
            step=1,
        )

        payment_count = st.number_input(
            "Payment Count",
            min_value=1,
            value=1,
            step=1,
        )

    with col2:
        seller_historical_problem_rate = st.slider(
            "Seller Historical Problem Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
        )

        category_historical_problem_rate = st.slider(
            "Category Historical Problem Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.12,
            step=0.01,
        )

        is_multi_seller_order = st.selectbox(
            "Is Multi Seller Order?",
            ["No", "Yes"],
        )

        customer_state = st.selectbox(
            "Customer State",
            ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "Other"],
        )

        seller_state = st.selectbox(
            "Seller State",
            ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "Other"],
        )

    input_data = {
        "customer_zip_code_prefix": 12345,
        "customer_state": customer_state,
        "item_count": 1,
        "product_count": 1,
        "seller_count": 1,
        "total_price": order_total_value,
        "total_freight_value": order_total_value * freight_ratio,
        "payment_type": "credit_card",
        "payment_installments_max": payment_count,
        "payment_count": payment_count,
        "total_payment_value": order_total_value,
        "customer_historical_problematic": 0,
        "customer_historical_orders": 3,
        "customer_historical_problem_rate": 0.05,
        "purchase_year": 2018,
        "purchase_month": 5,
        "purchase_dayofweek": 2,
        "purchase_hour": 14,
        "is_weekend": 0,
        "is_night_order": 0,
        "order_total_value": order_total_value,
        "freight_ratio": freight_ratio,
        "avg_item_price": order_total_value,
        "is_multi_item_order": 0,
        "is_multi_seller_order": 1 if is_multi_seller_order == "Yes" else 0,
        "is_installment_payment": 1 if payment_count > 1 else 0,
        "estimated_delivery_days": estimated_delivery_days,
        "approval_delay_hours": 2,
        "seller_state_mode": seller_state,
        "seller_city_nunique": 1,
        "seller_state_nunique": 1,
        "same_customer_seller_state": 1 if customer_state == seller_state else 0,
        "main_product_category": "bed_bath_table",
        "category_count": 1,
        "is_unknown_category": 0,
        "total_product_weight_g": 500,
        "avg_product_weight_g": 500,
        "max_product_weight_g": 500,
        "total_product_volume_cm3": 1000,
        "avg_product_volume_cm3": 1000,
        "max_product_volume_cm3": 1000,
        "avg_product_photos_qty": 3,
        "avg_product_description_length": 500,
        "seller_historical_problem_count": int(seller_historical_problem_rate * 10), # 5,
        "seller_historical_order_count": 50,
        "seller_historical_problem_rate": seller_historical_problem_rate,
        "category_historical_problem_count": 10,
        "category_historical_order_count": 100,
        "category_historical_problem_rate": category_historical_problem_rate,
        "seller_total_order_count": 50,
        "is_first_order": 0,
        "is_high_risk_seller": 1 if seller_historical_problem_rate >= 0.10 else 0,
        "is_high_risk_category": 1 if category_historical_problem_rate >= 0.10 else 0,
        "is_new_seller": 0,
        "heavy_multiitem_order": 0,
        "risky_long_delivery": 1 if estimated_delivery_days >= 10 else 0,
        "payment_type_count": 1,
        "has_voucher": 0,
        "has_boleto": 0,
        "shipping_limit_days": 3,
        "is_urgent_shipping": 0
    }

    st.markdown("---")

    if st.button("Predict Order Risk", type="primary"):

        result = call_prediction_api(input_data)

        if result is not None:
            # Prediction 
            # probability, prediction = result
            # risk_level = get_risk_level(probability)
            # action = get_recommended_action(risk_level)
            probability = result["risk_probability"]
            prediction = result["prediction"]
            risk_level = result["risk_level"]
            action = result["recommended_action"]

            st.markdown("---")
            st.markdown("### Prediction Result")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Risk Probability", f"{probability:.0%}")
            
            col2.metric(
                "Decision",
                "Manual Review" if prediction == 1 else "Monitor"
            )
            col3.metric("Risk Level", risk_level)
            col4.metric("Recommended Action", action)

            st.markdown("### Business Interpretation")

            if risk_level in ["High", "Critical"]:
                st.warning("Bu sipariş model tarafından sorunlu sipariş adayı olarak işaretlendi. "
                        "Operasyon ekibi bu siparişi öncelikli kontrol etmelidir.")
            elif risk_level == "Medium":
                st.warning("Bu siparişte risk sinyalleri tespit edildi. Takip önerilir.")
            else:
                st.success("Bu sipariş model tarafından düşük riskli olarak değerlendirildi. "
                        "Standart operasyon akışıyla devam edebilir.")

            st.markdown("### AI Explainability (SHAP)")

            try:
                pipeline, preprocessor, model, explainer, feature_names, raw_feature_columns = load_shap_artifacts()

                input_df = pd.DataFrame([input_data])
                input_df = input_df[raw_feature_columns]

                X_processed = preprocessor.transform(input_df)

                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()

                X_processed = pd.DataFrame(
                    X_processed,
                    columns=feature_names
                )

                shap_values = explainer.shap_values(X_processed)

                if isinstance(shap_values, list):
                    sv = shap_values[1]
                else:
                    sv = shap_values

                shap_importance = pd.DataFrame({
                    "feature": feature_names,
                    "impact": np.abs(sv[0])
                }).sort_values("impact", ascending=False).head(5)

                for _, row in shap_importance.iterrows():
                    st.info(
                        f"{prettify_feature_name(row['feature'])} → SHAP impact: {row['impact']:.3f}"
                    )

            except Exception as e:
                st.error(f"SHAP açıklaması üretilemedi: {e}")

                
            st.markdown("### AI-style Explanation")

            if risk_level == "Low":
                st.info(
                    "Bu sipariş düşük riskli görünüyor. Mevcut sinyaller operasyonel olarak kritik bir problem göstermiyor. "
                    "Standart sipariş akışıyla devam edilebilir."
                )

            elif risk_level == "Medium":
                st.warning(
                    "Bu siparişte bazı operasyonel risk sinyalleri bulunuyor. Teslimat süresi, satıcı geçmişi ve kargo oranı "
                    "takip edilmeli. Şimdilik manuel inceleme zorunlu değil, ancak monitoring önerilir."
                )

            elif risk_level == "High":
                st.error(
                    "Bu sipariş yüksek riskli görünüyor. Operasyon ekibi satıcı doğrulaması, teslimat takibi ve gerekirse "
                    "müşteri bilgilendirmesi yapmalıdır."
                )

            else:
                st.error(
                    "Bu sipariş kritik risk seviyesinde. Sipariş ilerlemeden önce manuel operasyon incelemesi önerilir."
                )

            st.markdown("### Recommended Operational Action")

            if risk_level == "Low":
                st.success(
                    """
                    Normal Flow

                    - Sipariş standart operasyon akışında ilerleyebilir.
                    - Ek manuel kontrol gerekmiyor.
                    - Rutin monitoring yeterlidir.
                    """
                )

            elif risk_level == "Medium":
                st.warning(
                    """
                    Monitor Order

                    - Teslimat zaman çizelgesi takip edilmeli.
                    - Satıcı geçmiş problem oranı kontrol edilmeli.
                    - Gecikme sinyali oluşursa operasyon ekibi bilgilendirilmeli.
                    """
                )

            elif risk_level == "High":
                st.error(
                    """
                    Priority Review

                    - Satıcıdan sipariş onayı alınmalı.
                    - Teslimat süreci yakından takip edilmeli.
                    - Gerekirse müşteri bilgilendirmesi yapılmalı.
                    """
                )

            else:
                st.error(
                    """
                    Manual Investigation Required

                    - Sipariş manuel incelemeye alınmalı.
                    - Satıcı, teslimat ve ödeme sinyalleri birlikte kontrol edilmeli.
                    - Operasyon yöneticisine eskalasyon önerilir.
                    """
                )

            st.markdown("### Top Risk Drivers")

            risk_drivers = []

            if freight_ratio >= 0.35:
                risk_drivers.append(
                    "High freight ratio: Kargo maliyetinin sipariş tutarına oranı yüksek."
                )

            if estimated_delivery_days >= 10:
                risk_drivers.append(
                    "Long estimated delivery time: Tahmini teslim süresi uzun."
                )

            if seller_historical_problem_rate >= 0.10:
                risk_drivers.append(
                    "Seller historical risk: Satıcının geçmiş problem oranı ortalamanın üzerinde."
                )

            if category_historical_problem_rate >= 0.10:
                risk_drivers.append(
                    "Category historical risk: Ürün kategorisinin geçmiş problem oranı yüksek."
                )

            if is_multi_seller_order == "Yes":
                risk_drivers.append(
                    "Multi-seller order: Sipariş birden fazla satıcı içeriyor."
                )

            if payment_count > 3:
                risk_drivers.append(
                    "Multiple payments: Ödeme/taksit yapısı operasyonel karmaşıklığı artırabilir."
                )

            if risk_drivers:
                for driver in risk_drivers:
                    st.warning(driver)
            else:
                st.success(
                    "No strong operational risk driver detected based on the current input."
                )
        



elif page == "Model Performance":
    st.title("📊 Model Performance")

    if not metrics_df.empty:
        st.subheader("Model Benchmark Results")
        st.dataframe(metrics_df, use_container_width=True)

        render_kpi_cards(metrics_df)

        best_model = get_best_model(metrics_df)

        if best_model is not None:
            st.markdown("### Model Result Interpretation")

            st.info(
                f"""
                En iyi model **{best_model.get("model_name")}** olarak seçildi.

                Bu seçim sadece accuracy'ye göre değil, operasyonel problem gereği
                **Recall** ve **F2-score** dikkate alınarak yapılmıştır.
                Çünkü bu projede riskli siparişleri kaçırmak, bazı normal siparişleri
                yanlışlıkla riskli işaretlemekten daha maliyetlidir.
                """
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Optimized Metrics")
                st.write(f"Recall: **{best_model.get('recall_optimized'):.4f}**")
                st.write(f"Precision: **{best_model.get('precision_optimized'):.4f}**")
                st.write(f"F1-score: **{best_model.get('f1_optimized'):.4f}**")
                st.write(f"F2-score: **{best_model.get('f2_optimized'):.4f}**")
                st.write(f"Best Threshold: **{best_model.get('best_threshold'):.2f}**")

            with col2:
                st.markdown("#### Business Reading")
                st.warning(
                    """
                    Model recall odaklı optimize edilmiştir.
                    Bu nedenle daha fazla riskli sipariş yakalanır;
                    fakat precision düşük kalabilir.

                    Bu trade-off operasyonel risk projelerinde kabul edilebilir,
                    çünkü amaç ilk aşamada potansiyel problemli siparişleri kaçırmamaktır.
                    """
                )

            st.markdown("### Overfitting Check")

            f2_gap = best_model.get("f2_gap")
            recall_gap = best_model.get("recall_gap")

            st.write(f"F2 Gap: **{f2_gap:.4f}**")
            st.write(f"Recall Gap: **{recall_gap:.4f}**")

            if f2_gap > 0.20:
                st.error(
                    """
                    Train ve test performansı arasında belirgin fark vardır.
                    Bu durum overfitting riskine işaret eder.

                    Bu proje için sonraki iterasyonda:
                    - SMOTE etkisi yeniden değerlendirilmeli,
                    - model complexity azaltılmalı,
                    - threshold tuning validasyon seti üzerinde yapılmalı,
                    - temporal validation güçlendirilmelidir.
                    """
                )
            else:
                st.success(
                    """
                    Train-test farkı kabul edilebilir seviyededir.
                    Model genelleme açısından daha stabil görünmektedir.
                    """
                )

        st.warning(
            "Bu skorlar leakage temizliği sonrası daha gerçekçi performansı temsil eder. "
            "Model iyileştirme ayrı bir iterasyon olarak planlanmıştır."
        )
    else:
        st.error("Model benchmark dosyası bulunamadı.")

    st.subheader("Cross Validation Results")

    if CV_PATH.exists():
        cv_df = pd.read_csv(CV_PATH)
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.info("Cross-validation dosyası bulunamadı.")


elif page == "Explainability":
    st.title("🔍 Explainability")

    st.markdown(
        """
        Bu sayfa modelin hangi değişkenlere göre risk tahmini yaptığını açıklamak için kullanılır.
        SHAP ve feature importance çıktıları, modelin karar mantığını daha şeffaf hale getirir.
        """
    )

    if SHAP_PATH.exists():
        image = Image.open(SHAP_PATH)
        max_width = 2000
        max_height = 520

        image.thumbnail((max_width, max_height))
        st.image(
            image,
            caption="SHAP Summary Plot"
        )
        
    else:
        st.error("SHAP görseli bulunamadı.")

    st.info(
        """
        SHAP analizi modelin hangi feature'lara göre risk kararı verdiğini gösterir.

        Pozitif yöndeki feature etkileri risk ihtimalini artırırken,
        negatif yöndeki etkiler risk ihtimalini azaltır.

        Bu ekran aynı zamanda leakage kontrolü için de kullanılmıştır.
        """
    )

    st.subheader("Top Feature Importance")

    if FEATURE_IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        importance_df["business_name"] = (
            importance_df["feature"]
            .apply(prettify_feature_name)
        )

        importance_df = importance_df.rename(
            columns={
                "feature": "technical_feature_name",
                "importance": "importance_score"
            }
        )

        st.dataframe(
            importance_df[
                [
                    "business_name",
                    "technical_feature_name",
                    "importance_score"
                ]
            ].head(20),
            use_container_width=True
        )
    else:
        st.error("Feature importance dosyası bulunamadı.")

    st.info(
        "Explainability çıktıları sadece sunum için değil, leakage kontrolü ve model güvenilirliği için de kullanılmıştır."
    )


elif page == "Business Actions":
    st.title("✅ Business Actions")

    st.markdown(
        """
        Model çıktısı tek başına yeterli değildir.
        Operasyon ekibinin anlayabileceği aksiyon önerisine çevrilmelidir.
        """
    )

    action_df = pd.DataFrame(
        {
            "Risk Level": ["Low", "Medium", "High", "Critical"],
            "Meaning": [
                "Order looks normal",
                "Some risk signals detected",
                "Strong risk signals detected",
                "Very high risk, manual review needed",
            ],
            "Recommended Action": [
                "Normal flow",
                "Monitor order",
                "Seller confirmation + delivery follow-up",
                "Manual review before order proceeds",
            ],
        }
    )

    st.dataframe(action_df, use_container_width=True)

    st.success(
        "Bu yapı projeyi teknik model çıktısından çıkarıp operasyonel karar destek ürününe dönüştürür."
    )


elif page == "AI Assistant":
    st.title("🤖 AI Assistant")
    st.caption("Data-aware assistant for operational analytics and project explanation")

    df = load_featured_data()

    assistant_mode = st.selectbox(
        "Assistant Mode",
        [
            "Local Data Assistant",
            "OpenAI Assistant",
            "Gemini Assistant",
        ],
    )

    question = st.text_input(
        "Ask a question:",
        placeholder="Example: Which categories have the highest risk?",
    )

    st.markdown("### Example Questions")
    st.code(
        """
            Which categories have the highest risk?
            Which sellers have the highest risky order rate?
            Which customer states are most risky?
            Does high freight ratio increase risk?
            Why is recall important in this project?
            Why did performance drop after leakage removal?
        """
    )

    if question:
        st.markdown("### Answer")

        try:
            if assistant_mode == "Local Data Assistant":
                answer = local_dataframe_answer(question, df)

                if isinstance(answer, pd.DataFrame):
                    st.dataframe(answer, use_container_width=True)
                else:
                    st.write(answer)

            elif assistant_mode == "OpenAI Assistant":
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("OPENAI_API_KEY .env içinde bulunamadı.")
                else:
                    answer = openai_answer(question, df)
                    st.write(answer)

            elif assistant_mode == "Gemini Assistant":
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("GEMINI_API_KEY .env içinde bulunamadı.")
                else:
                    answer = gemini_answer(question, df)
                    st.write(answer)

        except Exception as e:
            st.error(f"Assistant çalışırken hata oluştu: {e}")

elif page == "Project Story":
    
    st.title("📖 Project Story")

    st.markdown(
        """
        ### 1. Problem

        E-commerce operasyonlarında bazı siparişler iptal, gecikme, müşteri memnuniyetsizliği
        veya operasyonel problem riski taşır. Bu proje, bu siparişleri erken aşamada
        tahmin etmeyi amaçlar.

        ### 2. Dataset

        Olist Brazilian E-Commerce veri seti kullanılmıştır.
        Veri; sipariş, müşteri, satıcı, ürün, ödeme, review ve teslimat bilgilerini içerir.

        ### 3. Label Engineering

        Veri setinde doğrudan gerçek iade etiketi olmadığı için `problematic_order`
        hedef değişkeni weak supervision yaklaşımıyla üretilmiştir.

        Birden fazla label stratejisi denendi:
        - **İlk yaklaşım:** Review score + teslimat gecikmesi + operasyonel sinyaller kombinasyonu (risk score)
        - **Sorun:** Review score label'ı domine etti, model yeterli sinyal üretemedi
        - **Final label:** Canceled / unavailable sipariş statusu — en net ve tahmin edilebilir operasyonel sinyal

        Bu iterasyon süreci label engineering'in veri biliminde ne kadar kritik olduğunu gösterdi.

        ### 4. Leakage Control

        Review score ve gerçek teslimat gecikmesi label üretiminde kullanılabilir;
        fakat model inputu olarak kullanılamaz.

        Bu nedenle post-order değişkenler feature setinden çıkarılmıştır.

        ### 5. Model Benchmark

        Birden fazla model recall ve F2-score odaklı karşılaştırılmıştır.
        Çünkü bu problemde riskli siparişleri kaçırmamak önceliklidir.

        ### 6. Explainability

        SHAP ve feature importance çıktılarıyla modelin hangi değişkenlere göre karar verdiği analiz edilmiştir.

        ### 7. Deployment Architecture

        Proje sadece notebook/model çıktısı olarak bırakılmamıştır.  
        Model servis edilebilir bir yapıya taşınmıştır.

        Kullanılan mimari:

        - **Streamlit:** Kullanıcı arayüzü ve ürün demosu
        - **FastAPI:** Model serving API layer
        - **PostgreSQL:** Prediction logging ve operasyonel kayıt
        - **Docker Compose:** Servislerin birlikte çalıştırılması
        - **Pipeline.joblib:** Preprocessor + model birleşik inference pipeline
        - **Decision threshold:** F2-score odaklı optimize edilmiş threshold

        Canlı akış:

        `Streamlit UI → FastAPI /predict → Pipeline → PostgreSQL prediction_logs → Prediction History`

        Bu yapı sayesinde model tahminleri sadece ekranda gösterilmez; aynı zamanda izlenebilir, loglanabilir ve operasyonel olarak takip edilebilir hale gelir.

        ### 8. Next Iteration

        Bu proje production-like MVP seviyesine getirilmiştir.  
        Sonraki iterasyonlarda:

        - Overfitting azaltma
        - SMOTE etkisini yeniden değerlendirme
        - Temporal validation güçlendirme
        - Feature selection / model complexity azaltma
        - Model monitoring ve drift tracking
        - Prediction history üzerinden performans izleme
        - Local LLM veya RAG tabanlı AI assistant entegrasyonu

        eklenebilir.
        """
    )


elif page == "Prediction History":
    st.title("🧾 Prediction History")
    st.caption("Recent prediction logs written by FastAPI into PostgreSQL")

    logs = call_prediction_history_api()

    if not logs:
        st.info("Henüz prediction log bulunamadı.")
    else:
        history_df = pd.DataFrame(logs)
        
        history_df["risk_probability"] = (
            history_df["risk_probability"] * 100
        ).round(2)

        history_df = history_df.rename(
            columns={
                "id": "Log ID",
                "risk_probability": "Risk Probability (%)",
                "prediction": "Prediction",
                "risk_level": "Risk Level",
                "recommended_action": "Recommended Action",
                "created_at": "Created At",
            }
        )
        #KPI
        total_logs = len(history_df)
        avg_risk = history_df["Risk Probability (%)"].mean()
        high_risk_count = history_df[
            history_df["Risk Level"].isin(["High", "Critical"])
        ].shape[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total_logs)
        col2.metric("Average Risk", f"{avg_risk:.2f}%")
        col3.metric("High Risk Predictions", high_risk_count)


        st.markdown("### Recent Predictions")

        display_df = history_df[
            [
                "Log ID",
                "Created At",
                "Risk Probability (%)",
                "Prediction",
                "Risk Level",
                "Recommended Action",
            ]
        ].copy()

        display_df["Prediction"] = display_df["Prediction"].map(
            {
                0: "Normal Order",
                1: "Problematic Order",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Prediction Input Details")

        for _, row in history_df.head(5).iterrows():
            with st.expander(f"Log ID {row['Log ID']} | {row['Risk Level']} | {row['Risk Probability (%)']}%"):
                st.json(row["input_data"])

        st.success(
            "Bu ekran, model tahminlerinin operasyonel olarak loglandığını ve izlenebilir hale geldiğini gösterir."
        )