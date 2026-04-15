from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import clean_data


_MODEL = None
_PIPELINE = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_artifacts() -> tuple[Any, Any]:
    global _MODEL, _PIPELINE
    if _MODEL is not None and _PIPELINE is not None:
        return _MODEL, _PIPELINE

    model_path = _repo_root() / "model" / "model.pkl"
    pipeline_path = _repo_root() / "model" / "pipeline.pkl"

    if not model_path.exists() or not pipeline_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Expected files at: "
            f"{model_path} and {pipeline_path}. "
            "Run `python src/train.py` (or `python train.py` from `src/`) to generate them."
        )

    _MODEL = joblib.load(model_path)
    _PIPELINE = joblib.load(pipeline_path)
    return _MODEL, _PIPELINE


def predict(data: dict) -> int:
    model, pipeline = _load_artifacts()

    df = pd.DataFrame([data])
    df = clean_data(df)

    transformed = pipeline.transform(df)
    pred = model.predict(transformed)

    return int(pred[0])