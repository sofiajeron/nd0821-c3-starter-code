import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.evaluation import inference


@pytest.fixture
def data_fixture():
    return pd.read_csv("data/clean_census.csv")


@pytest.fixture
def train_data_fixture(data_fixture):
    train, _ = train_test_split(data_fixture, test_size=0.20)
    return train


@pytest.fixture
def test_data_fixture(data):
    _, test = train_test_split(data, test_size=0.20)
    return test


@pytest.fixture
def cat_features_fixture():
    cat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat


@pytest.fixture
def process_data_train_fixture(train_data_fixture, cat_features_fixture):
    X, y, _, _ = process_data(
        train_data_fixture, categorical_features=cat_features_fixture, label="salary", training=True)
    return X, y


@pytest.fixture
def encoder_lb_fixture(train_data_fixture, cat_features_fixture):
    _, _, encoder, lb = process_data(
        train_data_fixture, categorical_features=cat_features_fixture, label="salary", training=True)
    return encoder, lb


@pytest.fixture
def process_data_test_fixture(test_data_fixture, cat_features_fixture, encoder_lb_fixture):
    encoder, lb = encoder_lb_fixture
    X, y, _, _ =  process_data(
        test_data_fixture, categorical_features=cat_features_fixture, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X, y


@pytest.fixture
def train_model_fixture(process_data_train_fixture):
    X_train, y_train = process_data_train_fixture
    model = train_model(X_train, y_train)
    return model


@pytest.fixture
def predictions_fixture(model, process_data_test_fixture):
    X_test, _ = process_data_test_fixture
    preds = inference(model, X_test)
    return preds
