from fastapi.testclient import TestClient

from app import main, model_loader


class _StubModel:
    """MLflow 레지스트리 없이도 테스트가 돌도록 하는 가짜 모델."""

    def predict(self, X):
        return ["추움"] * len(X)


def setup_module(module):
    model_loader._model = _StubModel()
    model_loader._model_info = {"run_id": "test", "model_type": "stub", "test_accuracy": 1.0}


client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_adds_umbrella_when_rain():
    r = client.post(
        "/predict", json={"temperature": -2, "humidity": 60, "precipitation": 3.0}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "추움"
    assert "우산" in data["items"]


def test_predict_no_umbrella_when_dry():
    r = client.post(
        "/predict", json={"temperature": -2, "humidity": 40, "precipitation": 0.0}
    )
    assert r.status_code == 200
    assert "우산" not in r.json()["items"]
