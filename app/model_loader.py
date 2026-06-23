"""MLflow 레지스트리에서 운영 모델과 모델 정보를 가져온다.

서비스는 모델을 '파일'이 아니라 '실험 결과(레지스트리의 한 버전)'로 다룬다.
config.MODEL_URI 의 별칭(champion)만 바꾸면 코드 수정 없이 모델 교체/롤백이 된다.
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from config import MLFLOW_TRACKING_URI, MODEL_URI

_model = None
_model_info = None


def load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model


def get_model_info():
    """현재 서비스에 붙은 모델의 메타데이터(run_id, 종류, 성능)를 반환."""
    global _model_info
    if _model_info is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            info = mlflow.models.get_model_info(MODEL_URI)
            run = MlflowClient().get_run(info.run_id)
            _model_info = {
                "run_id": info.run_id,
                "model_type": run.data.params.get("model_type"),
                "test_accuracy": run.data.metrics.get("test_accuracy"),
            }
        except Exception:
            _model_info = {"run_id": "unknown", "model_type": None, "test_accuracy": None}
    return _model_info
