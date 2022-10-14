"""
Function that sends a request to the MLflow model REST API to get score predictions.
"""

import requests

MODEL_API_URL = 'https://place-de-marche-ml.herokuapp.com/predict'


def predict(json_app: {str}):
    try:
        response = requests.post(
            url=MODEL_API_URL,
            headers={'content-type': 'application/json'},
            params={"json_app": json_app}
        )
        response.raise_for_status()
        output = [response.json()['new_apps_prediction'],
                  response.json()['shap_values'],
                  response.json()['expectation_values'],
                  ]

    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output


if __name__ == '__main__':
    # Example of a nerd joke
    print("it is easier to be a geek at 24 than 42")
