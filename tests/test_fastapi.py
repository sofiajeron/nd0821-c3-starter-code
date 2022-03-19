def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the API that predicts salary from census data"}


def test_predict_1(client, json_sample_more_50k):
    response = client.post("/predict", json=json_sample_more_50k)
    assert response.status_code == 200
    assert response.json()['result'] == 1


def test_predict_0(client, json_sample_less_50k):
    response = client.post("/predict", json=json_sample_less_50k)
    assert response.status_code == 200
    assert response.json()['result'] == 0


def test_predict_error(client, json_sample_error):
    response = client.post("/predict", json=json_sample_error)
    assert response.status_code == 422