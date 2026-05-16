print("MAIN.PY ÇALIŞTI")

"""
Ana pipeline çalıştırma dosyası.
"""
from src.logger import get_logger
from src import data_loading
from src import label_engineering
from src import eda
from src import feature_engineering
from src import preprocessing
from src import model_training
from src import model_explainability

logger = get_logger(__name__)

def main():
    """
    Tüm ML pipeline'ını sırasıyla çalıştırır.
    """
    logger.info("=" * 80)
    logger.info("Problematic Order Risk Prediction started") # Customer Order Risk Scoring System
    logger.info("=" * 80)

    data_loading.run()
    label_engineering.run()
    eda.run()
    feature_engineering.run()
    preprocessing.run()
    model_training.run()
    model_explainability.run()

    logger.info("=" * 80)
    logger.info("BAŞARIYLA TAMAMLANDI")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()