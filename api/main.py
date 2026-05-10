print("MAIN.PY ÇALIŞTI")

"""
Ana pipeline çalıştırma dosyası.
"""
from src.logger import get_logger
from src import data_loading
from src import label_engineering
from src import feature_engineering

logger = get_logger(__name__)


def main():
    """
    Tüm ML pipeline'ını sırasıyla çalıştırır.
    """
    logger.info("=" * 80)
    logger.info("RETURN RISK PREDICTION PIPELINE BAŞLADI")
    logger.info("=" * 80)

    data_loading.run()
    label_engineering.run()
    feature_engineering.run()

    logger.info("=" * 80)
    logger.info("PIPELINE BAŞARIYLA TAMAMLANDI ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()