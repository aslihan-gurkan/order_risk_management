"""
Merkezi loglama modülü.
Tüm proje bu modülden logger alır.
print() kullanılmaz, her şey logger ile yazılır.
"""
import logging
import os
from datetime import datetime
from src.config import LOGS_PATH, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    İsme göre logger oluşturur ve döner.
    Hem terminale hem log dosyasına yazar.

    Args:
        name: Modül adı — her dosyada __name__ ile çağrılır

    Returns:
        Yapılandırılmış logger nesnesi

    Kullanım:
        from src.logger import get_logger
        logger = get_logger(__name__)
        logger.info("İşlem başladı")
        logger.error("Hata oluştu")
    """
    os.makedirs(LOGS_PATH, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Aynı logger'a birden fazla handler eklenmesini önle
    if logger.handlers:
        return logger

    # Format: zaman | seviye | modül | mesaj
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1) Terminal'e yaz
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2) Dosyaya yaz — her gün yeni dosya
    log_file = os.path.join(
        LOGS_PATH,
        f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger