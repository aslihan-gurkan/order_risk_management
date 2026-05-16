@st.cache_resource
def load_shap_artifacts():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_names = pd.read_csv(
        FEATURE_NAMES_PATH
    ).iloc[:,0].tolist()

    #explainer = shap.TreeExplainer(model)
    #explainer = shap.TreeExplainer(model, model_output="raw")
    """if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model.get_booster())
    else:
        explainer = shap.TreeExplainer(model)"""
    
    # XGBoost 3.x base_score format fix
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        config = booster.save_config()
        import json
        config_dict = json.loads(config)
        try:
            base_score_str = config_dict['learner']['learner_model_param']['base_score']
            base_score = float(base_score_str.strip('[]'))
            booster.set_param('base_score', str(base_score))
        except Exception:
            pass
        explainer = shap.TreeExplainer(booster)
    else:
        explainer = shap.TreeExplainer(model)

    return model, preprocessor, explainer, feature_names

HISTORY_API_URL = os.getenv(
    "HISTORY_API_URL",
    "http://api:8000/prediction-history"
)




@st.cache_resource
def load_shap_artifacts():
    pipeline = joblib.load(BASE_DIR / "models" / "pipeline.joblib")
    raw_feature_columns = joblib.load(BASE_DIR / "models" / "raw_feature_columns.joblib")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = pd.read_csv(FEATURE_NAMES_PATH).iloc[:, 0].tolist()

    explainer = shap.TreeExplainer(model)

    return pipeline, preprocessor, model, explainer, list(feature_names), raw_feature_columns
