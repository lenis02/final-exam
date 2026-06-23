"""FastAPI 서비스: 날씨를 받아 의류를 추천한다.

실행:  uvicorn app.main:app --reload
"""
import pandas as pd
from fastapi import FastAPI

from app import model_loader
from app.prediction_logger import log_prediction
from app.schemas import Recommendation, WeatherInput
from config import CLOTHING_MAP, RAIN_ITEM, RAIN_THRESHOLD
from ml.data_loader import FEATURES

app = FastAPI(title="날씨 기반 의류 추천 시스템")


def build_items(label: str, precipitation: float) -> list:
    """예측 구간(label) + 강수 여부로 최종 추천 리스트를 만든다."""
    items = list(CLOTHING_MAP.get(label, []))
    if precipitation > RAIN_THRESHOLD:
        items.append(RAIN_ITEM)
    return items


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Recommendation)
def predict(weather: WeatherInput):
    model = model_loader.load_model()

    X = pd.DataFrame(
        [[weather.temperature, weather.humidity, weather.precipitation]],
        columns=FEATURES,
    )
    label = str(model.predict(X)[0])

    items = build_items(label, weather.precipitation)
    info = model_loader.get_model_info()

    log_prediction(weather.model_dump(), label, items, info)
    return Recommendation(label=label, items=items, model_info=info)
