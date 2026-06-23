"""사용자 입력(날씨)과 추천 결과를 JSON Lines 로 기록.

쌓인 로그는 추후 데이터 드리프트 모니터링 / 재학습 트리거의 근거가 된다.
"""
import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_PATH = os.path.join(LOG_DIR, "predictions.log")


def log_prediction(weather: dict, label: str, items: list, model_info: dict | None = None):
    """한 건의 예측을 로그 파일에 한 줄(JSON)로 남기고, 기록한 dict를 반환."""
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": weather,
        "label": label,
        "items": items,
        "model_info": model_info,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
