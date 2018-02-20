from flask import Flask, jsonify, request
import fall_risk

app = Flask(__name__)

@app.route("/v1/")
def hello():
    return "Fall risk prediction API (logistic regression multivariate classifier)"

# This route is here for the admin to train the model the first time
@app.route('/v1/train_logistic', methods=['GET'])
def train():
    fall_risk.logistic_model()
    return "Logistic Regression Model Trained"

@app.route('/v1/predict', methods=['GET'])
def predict_fall_risk():
    age = float(request.args.get('age'))
    berg = float(request.args.get('berg'))
    gait = float(request.args.get('gait'))

    return jsonify(fall_risk.predict(age, berg, gait))

if __name__ == '__main__':
    app.run(debug=True)