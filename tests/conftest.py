import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from main import app
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model
from starter.starter.ml.evaluation import inference


@pytest.fixture(scope='session')
def data_fixture():
    return pd.read_csv("starter/data/clean_census.csv")


@pytest.fixture(scope='session')
def train_data_fixture(data_fixture):
    train, _ = train_test_split(data_fixture, test_size=0.20)
    return train


@pytest.fixture(scope='session')
def test_data_fixture(data_fixture):
    _, test = train_test_split(data_fixture, test_size=0.20)
    return test


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def process_data_train_fixture(train_data_fixture, cat_features_fixture):
    X, y, _, _ = process_data(
        train_data_fixture, categorical_features=cat_features_fixture, label="salary", training=True)
    return X, y


@pytest.fixture(scope='session')
def encoder_lb_fixture(train_data_fixture, cat_features_fixture):
    _, _, encoder, lb = process_data(
        train_data_fixture, categorical_features=cat_features_fixture, label="salary", training=True)
    return encoder, lb


@pytest.fixture(scope='session')
def process_data_test_fixture(test_data_fixture, cat_features_fixture, encoder_lb_fixture):
    encoder, lb = encoder_lb_fixture
    X, y, _, _ =  process_data(
        test_data_fixture, categorical_features=cat_features_fixture, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X, y


@pytest.fixture(scope='session')
def train_model_fixture(process_data_train_fixture):
    X_train, y_train = process_data_train_fixture
    model = train_model(X_train, y_train)
    return model


@pytest.fixture(scope='session')
def predictions_fixture(train_model_fixture, process_data_test_fixture):
    X_test, _ = process_data_test_fixture
    preds = inference(train_model_fixture, X_test)
    return preds


@pytest.fixture(scope='session')
def client():
    with TestClient(app) as cl:
        yield cl


@pytest.fixture(scope='session')
def json_sample_less_50k():
    payload = {
        'age': 19,
        'workclass': 'Private',
        'fnlgt': 168294,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Craft-repair',
        'relationship': 'Own-child',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }

    return payload


@pytest.fixture(scope='session')
def json_sample_more_50k():
    payload = {
        'age': 50,
        'workclass': 'Private',
        'fnlgt': 2000,
        'education': 'HS-grad',
        'education-num': 14,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 10000,
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States',
    }

    return payload

@pytest.fixture(scope='session')
def json_sample_error():
    payload = {
        'age': 'Not available',
        'workclass': 'Local-gov',
        'fnlgt': 12285,
        'education': 'HS-grad',
        'education-num': 12,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'Asian-Pac-Islander',
        'sex': 'Male',
        'capital-gain': 10000,
        'capital-loss': 1000,
        'hours-per-week': 80,
        'native-country': 'United-States'
    }

    return payload

