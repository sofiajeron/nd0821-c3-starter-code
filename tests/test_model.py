import xgboost
from starter.starter.ml.model import train_model


def test_train_model(process_data_train_fixture):
    X_train, y_train = process_data_train_fixture
    model = train_model(X_train, y_train)

    assert isinstance(model, xgboost.core.Booster)
