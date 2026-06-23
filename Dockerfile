# 날씨 기반 의류 추천 서비스 (FastAPI)
FROM python:3.12-slim

WORKDIR /app

# 1) 의존성 먼저 설치 (코드보다 앞에 두어 레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) 애플리케이션 코드 복사
COPY config.py .
COPY app/ ./app/
COPY ml/ ./ml/

# Render 등 PaaS 는 $PORT 환경변수로 포트를 주입한다 (기본 8000)
ENV PORT=8000
EXPOSE 8000

# 모델은 MLFLOW_TRACKING_URI(원격 서버)에서 런타임에 로드한다.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
