# 기말 프로젝트 보고서 — 날씨 기반 의류 추천 시스템 (MLOps)

> 표기 규칙: `⬜` 는 **직접 채워 넣어야 하는 항목**(URL, 스크린샷 등).

---

## 1. 프로젝트 개요
- **프로젝트 이름**: 날씨 기반 의류 추천 시스템 (weather-clothing-recommender)
- **프로젝트 목적**: `(온도, 습도, 강수량)` 입력으로 그날 입을 의류를 추천하는 서비스에
  MLOps 파이프라인(코드 변경 → 자동 테스트 → 학습 → MLflow 등록 → 서비스 반영 → 운영 로그/롤백)을
  적용해, 코드 수정 없이 모델을 교체·롤백할 수 있는 운영 체계를 구축한다.
- **GitHub 주소 (public)**: ⬜ (예: `https://github.com/<계정>/weather-clothing-recommender`)
- **배포 주소 및 캡쳐**: ⬜ Render URL (예: `https://weather-clothing.onrender.com`) + 화면 캡쳐
- **MLflow Tracking Server 주소 및 캡쳐**: ⬜ ngrok https URL + MLflow UI 캡쳐

## 2. 소프트웨어 주요 기능 (서비스 / ML 분리)
**1) 사용자 핵심 기능 (서비스)**
- `POST /predict` : 날씨를 입력하면 추천 의류 리스트를 반환
- `GET /health` : 서비스 상태 확인(헬스체크)

**2) ML 모델이 사용되는 위치**
- `app/model_loader.py` : MLflow 레지스트리(`models:/weather-model@champion`)에서 모델을 로드(최초 1회 캐시)
- `app/main.py` 의 `/predict` : 로드된 모델로 체감 온도 구간을 예측

**3) 입력 데이터와 출력 결과**
- 입력: `{ "temperature": float, "humidity": 0~100, "precipitation": >=0 }`
- 출력:
```json
{
  "label": "추움",
  "items": ["패딩", "목도리", "기모바지", "장갑", "우산"],
  "model_info": {"run_id": "...", "model_type": "RandomForestClassifier", "test_accuracy": 0.9375}
}
```
- **서비스 로직 vs ML 로직 분리**: 구간 예측(label)은 **ML 모델**, 구간→의류 매핑과 강수 시 "우산"
  추가는 **서비스 규칙**(`config.CLOTHING_MAP`, `app/main.build_items`)으로 분리.

## 3. 실행 환경
- **OS**: Windows 11 (10.0.26200)
- **Git/GitHub**: git 2.47.0.windows.1 / GitHub(public 저장소) + GitHub Actions
- **Docker**: Docker 29.4.0 (python:3.12-slim 베이스 이미지)
- **MLflow**: mlflow 3.10.1 (Tracking + Model Registry), backend-store `sqlite:///mlflow.db`
- **Python/패키지**: conda env `weather-clothing` (Python 3.12.13), scikit-learn 1.8.0, FastAPI 0.128.0
- **배포 환경**: Render (Docker runtime, free plan) + 로컬 MLflow 서버를 ngrok 으로 노출

## 4. 전체 MLOps 파이프라인 구조
- **코드 변경 흐름**: 로컬 수정 → 커밋 → `git push origin main`
- **모델 학습 흐름**: push → GitHub Actions가 `pytest` 통과 시 `python -m ml.train` 실행
- **모델 등록/반영 흐름**: 학습이 MLflow에 run 기록 + `weather-model` 새 버전 등록 → 성능 확인 후
  `champion` 별칭을 신규 버전으로 이동(수동/반자동) → 서비스는 재배포 없이 새 모델 사용
- **서비스 운영 흐름**: Render의 FastAPI가 요청마다 champion 모델로 추천 + `logs/predictions.log` 기록
```
[Service/ML Code] --push--> [GitHub Actions: test→train] --log/register--> [MLflow(ngrok)]
                                                                                 │ champion alias
                                                                                 ▼
                                                          [Render FastAPI] --예측/로깅--> [LOG]
```

## 5. Git 기반 개발 과정
- **개발 흐름**: 골격(config/의존성) → 데이터·학습(ml) → 서비스(app) → 테스트 → CI → 컨테이너/배포 순으로,
  기능이 동작 가능한 단위로 점진적 구현.
- **커밋 전략**: Conventional Commits(`chore/feat/test/ci/build` + scope). 1커밋=1논리변경, 메시지는
  "무엇을/왜"를 본문에 기재. 실제 히스토리:
  ```
  chore: scaffold project structure and configuration
  feat(ml): add dataset and RandomForest training with MLflow
  feat(app): add FastAPI clothing-recommendation service
  test: add training and API tests
  ci: add GitHub Actions auto-training workflow
  build: containerize service and add Render deploy config
  ```
- **브랜치 사용**: 단독 개발이라 `main` 트렁크 기반으로 진행(학습 자동화도 `main` push 트리거).
  기능 규모가 커지면 `feat/*` 브랜치 + PR 전략으로 확장 가능.

## 6. CI/CD 구성 (`.github/workflows/train.yml`)
- **트리거**: `push`(paths: `ml/`,`app/`,`tests/`,`config.py`,`requirements.txt`,워크플로) + 수동 `workflow_dispatch`
- **자동화 범위**: 테스트 → (통과 시) 재학습 → 학습 산출물 업로드. *DevOps의 `push→test`에 MLOps의 `train→register`를 결합.*
- **주요 단계**: ① checkout ② setup-python 3.12 ③ `pip install -r requirements.txt` ④ `pytest -q`
  ⑤ `python -m ml.train` ⑥ `upload-artifact (ml/artifacts/)`
- **핵심 설계**: 테스트가 깨지면 학습이 실행되지 않음 → 깨진 코드로 모델이 갱신되는 사고 방지.
- **실행 결과 캡쳐**: ⬜ Actions 실행 성공 화면

## 7. Docker 기반 환경 구성
- **Dockerfile 주요 설정**: `python:3.12-slim` → requirements 먼저 설치(레이어 캐시) → 앱 코드 복사 →
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT` 로 기동. 모델은 이미지에 넣지 않고 런타임에
  `MLFLOW_TRACKING_URI`(원격)에서 로드.
- **실행 방법**:
  ```bash
  docker build -t weather-clothing:latest .
  docker run -p 8000:8000 \
    -e MLFLOW_TRACKING_URI="https://<ngrok>.ngrok-free.dev" \
    -e MODEL_URI="models:/weather-model@champion" \
    weather-clothing:latest
  # http://localhost:8000/health , POST /predict
  ```

## 8. ML 모델 구성
- **사용 데이터**: `ml/data/weather.csv` (240행). 컬럼 `temperature, humidity, precipitation, label`.
  체감온도 규칙 + 약 3% 라벨 노이즈로 생성(결정적 seed=42).
- **모델 종류**: `RandomForestClassifier(n_estimators=100, random_state=42)` (scikit-learn)
- **학습 코드 설명(`ml/train.py`)**: 데이터 로드 → `train_test_split(0.2, stratify)` →
  `train_model()` 학습 → MLflow에 params/metrics/artifacts/model 기록 → `weather-model` 등록.
- **평가 지표**: 정확도(accuracy) — train/test 두 가지 기록.
- **초기 모델 vs 신규 모델 비교**:

  | 버전 | 모델 | 변경점 | test_accuracy |
  |------|------|--------|---------------|
  | v1 (초기) | RandomForest(100) | 최초 학습 | 0.9375 |
  | v2 (신규) | ⬜ | ⬜ (데이터/파라미터 변경) | ⬜ |

## 8(b). MLflow 기반 실험 관리
- **MLflow Tracking 사용**: 예 (`mlflow.set_tracking_uri` + `start_run`)
- **기록 항목**:
  - parameter: `model_type, n_estimators, features, data_path, train_row_count, test_row_count`
  - metric: `train_accuracy, test_accuracy`
  - artifact: 학습 데이터 CSV, `weather_model.joblib`
  - model: `mlflow.sklearn.log_model(..., registered_model_name="weather-model")`
- **실험 결과 / 화면 캡쳐**: ⬜ MLflow Experiments·Runs·Model registry 캡쳐
- **가장 좋은 모델 선정 기준**: 동일 테스트셋 기준 `test_accuracy` 최고값 → 해당 버전에 `champion` 부여.

## 9. 모델 등록 및 서비스 반영
- **모델 저장 방식**: MLflow Model Registry에 `weather-model` 로 버전 관리(아티팩트는 서버 store에 저장).
- **서비스 로드 방식**: `app/model_loader.load_model()` 이 `models:/weather-model@champion` 을
  `mlflow.sklearn.load_model` 로 불러와 캐시.
- **신규 모델 반영 방법**: MLflow UI(또는 API)에서 `champion` 별칭을 신규 버전으로 이동 → 서비스 재시작 시 반영.
- **자동/수동 선택 이유**: 학습·등록은 **자동**(CI), 운영 반영(champion 교체)은 **수동/반자동**.
  잘못된 모델이 자동으로 운영에 올라가는 위험을 막고, 성능 비교 후 사람이 승인하는 게 안전하기 때문.

## 10. 재학습 / 모델 개선 과정
- **재학습 이유/방법**: ⬜ (예: 데이터 추가 또는 하이퍼파라미터 변경 후 `python -m ml.train` 재실행)
- **무엇이 바뀌었나(데이터/코드/파라미터)**: ⬜
- **재학습 전·후 성능 비교**: ⬜ (v1 0.9375 → v2 ⬜)
- **모델 교체 결과**: ⬜ (champion을 v2로 이동 후 서비스 응답의 `model_info` 변화 캡쳐)

## 11. 운영 로그 및 문제 대응
- **서비스/예측 요청 로그**: `app/prediction_logger.py` 가 요청마다 `logs/predictions.log` 에 JSONL 기록.
  실제 기록 예:
  ```json
  {"timestamp":"2026-06-23T20:34:28","input":{"temperature":-2.0,"humidity":60.0,"precipitation":3.0},
   "label":"추움","items":["패딩","목도리","기모바지","장갑","우산"],
   "model_info":{"run_id":"c0c23d1c...","model_type":"RandomForestClassifier","test_accuracy":0.9375}}
  ```
- **모델 정보 확인**: 응답·로그의 `model_info`(run_id/model_type/test_accuracy)로 어떤 모델이 응답했는지 추적.
- **일부러 발생시킨 문제 / 원인 / 해결**: ⬜ (계획: MLflow 서버를 내린 상태로 `/predict` 호출 →
  모델 로드 실패로 500 또는 `run_id="unknown"`. 원인=원격 트래킹 서버 미가용, 해결=서버/ngrok 재기동.
  → 14번에 상세 기재) — 아래 14번의 실제 사례도 참고.

## 12. 롤백 및 이전 모델 관리
- **이전 모델 보관**: MLflow Registry가 모든 버전(v1, v2, ...)을 영구 보관.
- **되돌리는 방법**: `champion` 별칭을 이전 버전으로 다시 이동(코드/재배포 불필요).
  ```python
  from mlflow.tracking import MlflowClient
  MlflowClient().set_registered_model_alias("weather-model", "champion", 1)  # v1로 롤백
  ```
- **버전 관리 화면/코드**: ⬜ Model registry의 Versions/Aliases 캡쳐.

## 13. 전체 파이프라인 동작 흐름
- **코드 수정 → 서비스 반영**: 코드 push → CI 통과 → (필요 시) 이미지 재배포 → 서비스 갱신.
- **데이터 변경 → 재학습**: `ml/data/weather.csv` 갱신 push → CI가 자동 재학습·신규 버전 등록.
- **모델 변경 → 운영 반영**: 신규 버전 성능 확인 → `champion` 별칭 이동 → 서비스가 재배포 없이 새 모델 사용.

## 14. 문제 해결 경험 (실제 발생)
1. **`environment.yml` 한글 주석 → conda charmap 디코드 경고**
   - 원인: Windows conda가 파일을 cp1252로 읽어 한글 바이트 디코드 실패.
   - 해결: 주석을 ASCII로 교체. (환경 생성 자체는 정상 완료)
2. **`conda run python -c "<여러 줄>"` 실패 (`NotImplementedError: arguments contain newlines`)**
   - 원인: conda run이 줄바꿈 포함 인자를 지원하지 않음.
   - 해결: 코드를 `.py` 파일로 분리해 env의 python으로 직접 실행.
3. **한글 예측 결과 출력 시 `UnicodeEncodeError: 'charmap'`**
   - 원인: PowerShell 콘솔 기본 인코딩(cp1252/cp949)으로 한글 stdout 인코딩 실패(파일 자체는 UTF-8 정상).
   - 해결: `PYTHONIOENCODING=utf-8` 설정 후 실행 / 로그는 `encoding="utf-8"`로 기록.
- ⬜ (선택) 추가 사례: 배포/서버 연동 중 겪은 문제 기재.

---

### ✅ 직접 채워야 할 항목 체크리스트
- [ ] GitHub public 저장소 생성 + URL (1번)
- [ ] Render 배포 + URL + 캡쳐 (1, 6번)
- [ ] 로컬 MLflow 서버 + ngrok URL + MLflow UI 캡쳐 (1, 8b번)
- [ ] GitHub Actions 실행 성공 캡쳐 (6번)
- [ ] 신규 모델(v2) 학습·비교·champion 교체 (8, 10, 12, 13번)
- [ ] Model registry 버전/별칭 캡쳐 (12번)
