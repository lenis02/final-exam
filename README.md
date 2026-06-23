# 날씨 기반 의류 추천 시스템 (MLOps 파이프라인)

`(온도, 습도, 강수량)` 을 입력받아 **체감 온도 구간**을 분류하고, 구간별 의류 + 강수 시 우산을
추천하는 서비스. 학습(MLflow) → 등록 → 서비스 로딩 → 예측 로깅의 MLOps 흐름을 따른다.

## 폴더 구조
```
weather-clothing-recommender/
├── config.py                 # MLflow URI / 모델 정보 / 추천 규칙 (단일 설정)
├── requirements.txt
├── environment.yml           # Anaconda 가상환경 정의
├── .github/workflows/train.yml   # CI: pytest -> 재학습 -> 모델 등록
├── app/                      # 서비스(추론) 코드
│   ├── main.py               # FastAPI 엔드포인트 (/predict, /health)
│   ├── model_loader.py       # MLflow 레지스트리에서 모델 로드
│   ├── prediction_logger.py  # 입력+추천결과 로깅
│   └── schemas.py            # 요청/응답 스키마(Pydantic)
├── ml/                       # 데이터 & 학습 코드
│   ├── train.py              # 학습 + MLflow 등록 (registered_model_name="weather-model")
│   ├── data_loader.py
│   └── data/weather.csv
├── logs/predictions.log      # 런타임 생성
└── tests/                    # test_api.py, test_train.py
```

## 1. Anaconda 가상환경 구성

```bash
# 방법 A) environment.yml 로 한 번에 생성
conda env create -f environment.yml
conda activate weather-clothing

# 방법 B) 직접 생성
conda create -n weather-clothing python=3.12 -y
conda activate weather-clothing
pip install -r requirements.txt
```

의존성을 바꾼 뒤 환경을 갱신하려면:
```bash
conda env update -f environment.yml --prune
```

## 2. 모델 학습 (MLflow 기록 + 등록)
```bash
python -m ml.train          # accuracy 기록 + weather-model 로 레지스트리 등록
mlflow ui --port 9999       # 실험/모델 확인 (http://127.0.0.1:9999)
```
운영용 모델 버전에 MLflow UI 에서 `champion` **별칭(alias)** 을 붙이면, `config.MODEL_URI`
(`models:/weather-model@champion`) 가 코드 수정 없이 그 모델을 가져온다. 문제 시 별칭만 이전
버전으로 옮기면 롤백된다.

## 3. 서비스 실행
```bash
uvicorn app.main:app --reload
# POST /predict  body: {"temperature": -2, "humidity": 60, "precipitation": 3.0}
```

## 4. 통합 로컬 실행 (Docker Compose)
MLflow 서버와 서비스를 한 번에 띄운다 (ngrok 없이 컨테이너끼리 통신).
```bash
docker compose up -d --build                                    # mlflow(5000) + app(8000)
MLFLOW_TRACKING_URI=http://localhost:5000 python -m ml.train    # 모델 등록 + champion 별칭
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"temperature":-2,"humidity":60,"precipitation":3.0}'  # -> 추움 + 우산
```
> 컨테이너 간 호출 차단(DNS rebinding 403)을 막기 위해 mlflow 서버에 `--allowed-hosts` 를 지정했다.

## 5. 테스트
```bash
pytest -q
```

## 6. CI/CD
`main` 브랜치 push(또는 수동 `workflow_dispatch`) 시 GitHub Actions 가
`pytest` → `python -m ml.train` → 학습 산출물 업로드를 자동 수행한다.
