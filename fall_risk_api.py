from flask import Flask, jsonify, request
from sklearn.externals import joblib
import fall_risk

app = Flask(__name__)

@app.route("/")
def hello():
    return "Fall risk prediction API (logistic regression multivariate classifier)"

# NEXT STEPS:
# 1) Pickle the model, so you only have to train once
# 2) Deploy on Heroku

@app.route('/predict', methods=['GET'])
def predict():
    age = float(request.args.get('age'))
    berg = float(request.args.get('berg'))
    gait = float(request.args.get('gait'))

    return jsonify(fall_risk.predict(age, berg, gait))

if __name__ == '__main__':
    app.run(debug=True)
