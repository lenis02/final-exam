"""프로젝트 전역 설정: MLflow 연결 정보와 모델 정보를 한 곳에서 관리.

학습 코드(ml/)와 서비스 코드(app/)가 동일한 모델 정보를 공유하도록 하는
단일 진실 공급원(Single Source of Truth) 역할을 한다.
"""
import os

# --- MLflow 설정 ---
# 로컬 실습 기본값은 sqlite. 서버를 쓰면 환경변수로 원격(ngrok 등) URL을 넣는다.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# 학습 실행(run)을 묶는 experiment 이름
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "weather-clothing-local")

# --- 모델 정보 ---
# 모델 레지스트리에 등록할 이름 (ml/train.py 에서 사용)
REGISTERED_MODEL_NAME = "weather-model"

# 서비스가 가져올 모델 URI.
# 운영용 모델에는 MLflow UI 에서 'champion' 별칭(alias)을 붙여서 사용한다.
#   - 특정 버전:    "models:/weather-model/1"
#   - 별칭(권장):   "models:/weather-model@champion"  (코드 수정 없이 모델 교체/롤백)
MODEL_URI = os.getenv("MODEL_URI", "models:/weather-model@champion")

# --- 추천 도메인 규칙 ---
# 분류 모델이 예측한 '체감 온도 구간(label)' -> 기본 의류 추천 리스트
CLOTHING_MAP = {
    "더움": ["반팔티", "반바지", "샌들"],
    "따뜻함": ["반팔티", "면바지", "운동화"],
    "선선함": ["긴팔티", "가디건", "청바지"],
    "쌀쌀함": ["니트", "자켓", "청바지"],
    "추움": ["패딩", "목도리", "기모바지", "장갑"],
}

# 강수가 있을 때 추가로 추천할 품목과 임계값(mm)
RAIN_ITEM = "우산"
RAIN_THRESHOLD = 0.0
