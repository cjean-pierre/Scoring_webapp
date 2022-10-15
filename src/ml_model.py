"""
Function that sends a request to the MLflow model REST API to get score predictions.
"""

import requests


def predict_score(json_app: str):

    model_api_url = 'https://pret-a-depenser-ml.herokuapp.com/predict'
    try:
        response = requests.post(
            url=model_api_url,
            headers={'content-type': 'application/json'},
            json=json_app
        )
        response.raise_for_status()
        output = response.json()['new_apps_prediction']

    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output


def predict_shap(json_app: str):
    model_api_url = 'https://pret-a-depenser-ml.herokuapp.com/shap'
    try:
        response = requests.post(
            url=model_api_url,
            headers={'content-type': 'application/json'},
            json=json_app
        )
        response.raise_for_status()
        output = [
                  response.json()['shap_values'],
                  response.json()['expectation_values'],
                  ]

    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output


if __name__ == '__main__':
    # Example of a nerd joke
    print("it is easier to be a geek at 24 than 42")
