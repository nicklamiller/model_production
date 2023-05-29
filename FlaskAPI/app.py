import json
import flask
import pickle
import rootpath

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():

    request_json = request.get_json()
    x = float(request_json['input'])

    model = load_models()
    prediction = model.predict([[x]])[0]

    response = json.dumps({'response': prediction})
    return response, 200


def load_models():
    path = rootpath.detect()

    file_name = f"{path}/models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


if __name__ == '__main__':
     application.run(debug=True)

