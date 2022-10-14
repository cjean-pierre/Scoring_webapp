"""
Function that sends a request to the MLflow model REST API to get score predictions.
"""

import requests
from fastapi import Body
import json

def predict_score(json_app: dict = Body({})):

    model_api_url = 'https://place-de-marche-ml.herokuapp.com/predict'
    try:
        response = requests.post(
            url=model_api_url,
            headers={'content-type': 'application/json'},
            params={"json_app": json.dumps(json_app)}
        )
        response.raise_for_status()
        output = response.json()['new_apps_prediction']

    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output


def predict_shap(json_app: dict = Body({})):
    model_api_url = 'https://place-de-marche-ml.herokuapp.com/shap'
    try:
        response = requests.post(
            url=model_api_url,
            headers={'content-type': 'application/json'},
            params={"json_app": json.dumps(json_app)}
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
