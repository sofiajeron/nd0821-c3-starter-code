import requests
import json

url = 'https://census-data-app.herokuapp.com/predict'

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

if __name__ == '__main__':
    request = json.dumps(payload)
    print(f"request: {request}")
    try:
        response = requests.post(url, data=request, headers={'result-type': 'application/json'})
        response.status_code == 200
        print(f"response status code: {response.status_code}")
        result = response.json()['result']
        print(f"prediction: {result}")

    except Exception as err:
        raise err
