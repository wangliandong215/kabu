"""
data_provider/logger.py — shared logging setup for data providers.

kabu's existing modules use print() for diagnostics; providers under
data_provider/ use the stdlib logging module instead so request timing and
errors are structured and level-filterable.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
