"""학습 데이터 로딩 / 전처리.

train.py 를 간결하게 유지하기 위해 데이터 적재 로직을 분리한다.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "weather.csv")

# 입력 피처와 정답(타깃) 컬럼
FEATURES = ["temperature", "humidity", "precipitation"]
TARGET = "label"


def load_data(data_path: str = DATA_PATH):
    """CSV를 읽어 (X, y, data_path)를 반환한다."""
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, data_path
