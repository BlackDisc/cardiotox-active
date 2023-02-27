import logging
import sys
import numpy as np
from joblib import load
from flask import Flask, flash, request, redirect, url_for, jsonify, make_response
from werkzeug.utils import secure_filename
from typing import List

sys.path.append("src/")
from features import extract_features_from_smile

app = Flask(__name__)

# Loading model
app.batch_model = load("model/batch_model.joblib")
app.logger.setLevel(logging.INFO)


@app.route("/")
def intro():
    app.logger.info("Service intro")
    return "CardioTox webservice"


@app.route("/predict", methods=["POST"])
def predict_cardiotox():
    app.logger.info(f"Number of smiles to predict:{len(request.get_json()['smiles'])}")

    predictions = smile_cardiotox_prediction(request.get_json()["smiles"])

    app.logger.info(f"Number of predicted smiles:{len(predictions)}")
    return make_response(jsonify({"predictions": predictions}), 200)


def smile_cardiotox_prediction(smiles: List[str]) -> List[int]:
    # Feature extraction
    features = extract_features_from_smile(
        smiles, feature_sets=["DESC"], desc_file_path="model/batch_model_features.txt"
    )

    # Model prediction
    return app.batch_model.predict(features).tolist()


if __name__ == "__main__":
    app.run()
