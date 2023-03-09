import os
from flask import Flask, request, json
from model import BakeryModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data/"


@app.before_first_request
def init():
    global model
    model = BakeryModel(timestep=5, predict_len=1, features=4)
    model.load_model()


@app.route('/bakery/predict', methods=['POST', 'GET'])
def predict():
    if model.model == None:
        response = {"result": "No pretrained model, please train a model first"}
    elif request.method == 'POST':
        data_info = request.form
        if data_info:
            if data_info.get("predict_variable"):
                predict_variable = data_info.get("predict_variable")
            if data_info.get("short_history"):
                predict_data = data_info.get("short_history")
                predict_data = eval(predict_data)
                prediction = model.predict(predict_data, predict_variable)
                result = {"result": [int(one) for one in prediction.tolist()]}
        response = result
        print(response)

    return json.dumps(response)


@app.route('/bakery/train', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        data_info = request.form
        file_info = request.files["file"]

        if data_info and file_info:
            features = eval(data_info.get("features"))
            model.features = len(features)
            file_name = file_info.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file_info.save(file_path)
            history = model.train_model(data_path=file_path, features=features, evaluate=False)
            result = {"history": history}
        response = result

    return json.dumps(response)


if __name__ == "__main__":
    app.run(debug=False, port=8082)