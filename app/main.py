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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body style="font-family: sans-serif; padding: 20px;">
            <h2>날씨 기반 의류 추천 시스템</h2>
            <form id="weatherForm">
                온도 (°C): <input type="number" id="temp" value="20" step="0.1"><br><br>
                습도 (%): <input type="number" id="hum" value="50" step="0.1"><br><br>
                강수량 (mm): <input type="number" id="rain" value="0" step="0.1"><br><br>
                <button type="button" onclick="predict()">추천 받기</button>
            </form>
            <div id="result" style="margin-top: 20px; border: 1px solid #ccc; padding: 10px;"></div>

            <script>
            async function predict() {
                const data = {
                    temperature: parseFloat(document.getElementById('temp').value),
                    humidity: parseFloat(document.getElementById('hum').value),
                    precipitation: parseFloat(document.getElementById('rain').value)
                };
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById('result').innerHTML = 
                    "<strong>결과:</strong> " + result.label + "<br>" +
                    "<strong>추천 의류:</strong> " + result.items.join(", ");
            }
            </script>
        </body>
    </html>
    """

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
