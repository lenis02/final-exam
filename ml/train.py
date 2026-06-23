"""날씨 -> 의류 구간 분류 모델 학습 및 MLflow 등록.

실행:  python -m ml.train
"""
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from ml.data_loader import FEATURES, load_data

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "weather_model.joblib")

# 하이퍼파라미터
N_ESTIMATORS = 100
RANDOM_STATE = 42


def train_model(X_train, y_train):
    """RandomForest 분류기를 학습해서 반환 (테스트에서도 재사용)."""
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X, y, data_path = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 실험 세팅
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 실험 기록 시작
    with mlflow.start_run():
        # 실험 설정(params) 기록
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("train_row_count", len(X_train))
        mlflow.log_param("test_row_count", len(X_test))

        model = train_model(X_train, y_train)

        # 성능 지표(metrics) 기록
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        # 로컬 파일로도 저장
        joblib.dump(model, MODEL_PATH)

        # artifact 기록 (데이터 + 모델 파일)
        mlflow.log_artifact(data_path)
        mlflow.log_artifact(MODEL_PATH)

        # MLflow 모델 형식으로 저장 + 모델 레지스트리에 등록
        mlflow.sklearn.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"Model saved to: {MODEL_PATH}")
        print(f"train_accuracy: {train_acc:.4f} | test_accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
