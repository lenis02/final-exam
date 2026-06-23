from ml.data_loader import FEATURES, load_data
from ml.train import train_model


def test_load_data():
    X, y, _ = load_data()
    assert len(X) == len(y)
    assert len(X) > 0
    assert list(X.columns) == FEATURES


def test_train_model_predicts():
    X, y, _ = load_data()
    model = train_model(X, y)
    preds = model.predict(X.head())
    assert len(preds) == 5
