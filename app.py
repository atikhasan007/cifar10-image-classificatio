from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import os

from src.cnnClassifier.pipeline.predict import PredictionPipeline
from src.cnnClassifier.utils.common import decodeImage


os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

app = Flask(__name__)
CORS(app)


class ClientAPP:

    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientAPP()


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train():

    os.system("python main.py")

    return "Training completed successfully."


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    image_data = request.json["image"]

    decodeImage(image_data, clApp.filename)

    result = clApp.classifier.predict()

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )