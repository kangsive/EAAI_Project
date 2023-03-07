import os
from flask import Flask, request, json
from model import BakeryModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "data/"


@app.before_first_request
def init():
    global model
    model = BakeryModel(timestep=4, predict_len=1, features=5)


@app.route('/bakery/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data_info = request.form
        if data_info:
            if data_info.get("short_history"):
                predict_data = data_info.get("short_history")
                # result = model.predict(predict_data)
                result = {"result": "predict result"}
        response = result
        # data["response"] = response

    return json.dumps(response)


@app.route('/bakery/train', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        data_info = request.form
        file_info = request.files["file"]

        if data_info and file_info:
            file_name = file_info.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file_info.save(file_path)
            # result = model.train_model()
            result = {"result": "train result"}
        response = result

    return json.dumps(response)


if __name__ == "__main__":
    app.run(debug=False, port=8082)