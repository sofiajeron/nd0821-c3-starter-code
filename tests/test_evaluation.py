from starter.ml.evaluation import compute_model_metrics, inference


def test_inference(model, process_data_test_fixture):
    X_test, _ = process_data_test_fixture
    preds = inference(model, X_test)

    assert preds.size[0] == X_test.shape[0]


def test_compute_model_metrics(process_data_test_fixture, predictions):
    _, y_test = process_data_test_fixture
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
