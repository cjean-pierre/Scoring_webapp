"""
Function that sends a request to the MLflow model REST API to get score predictions.
"""

import requests

MODEL_API_URL = 'https://fastapiroku-ml.herokuapp.com/predict'


def predict():
    try:
        response = requests.post(
            url=MODEL_API_URL,
            headers={'content-type': 'application/json'},
            # params={"x1": x1, "x2": x2, "x3": x3, "x4": x4},
        )
        response.raise_for_status()
        output = [response.json()['new_apps_file'],
                  response.json()['new_apps_prediction'],
                  response.json()['shap_values'],
                  response.json()['expectation_values'],
                  response.json()['shap_summary']
                  ]

    except (requests.HTTPError, IOError) as err:
        output = str(err)
    return output


if __name__ == '__main__':
    # Example of a nerd joke
    print("it is easier to be a geek at 24 than 42")
