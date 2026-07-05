"""
engine/_finbert_cache.py — Singleton cache for FinBERT pipeline.

Only loaded if 'transformers' and 'torch' are installed.
Model: ProsusAI/finbert (400 MB, downloads once to ~/.cache/huggingface/)
"""
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _pipeline
