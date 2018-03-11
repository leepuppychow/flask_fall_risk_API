from flask import Flask, jsonify, request
from flask_cors import CORS
import fall_risk
import low_back_pain

app = Flask(__name__)
CORS(app)

@app.route("/v1/")
def hello():
    return "Fall risk prediction API (logistic regression multivariate classifier)"

@app.route('/v1/predictfallrisk', methods=['GET'])
def predict_fall_risk():
    age = float(request.args.get('age'))
    berg = float(request.args.get('berg'))
    gait = float(request.args.get('gait'))

    return jsonify(fall_risk.predict(age, berg, gait))

@app.route('/v1/predictlowbackpain', methods=['GET'])
def predict_LBP():
    pelvic = float(request.args.get('pelvic'))
    lumbar = float(request.args.get('lumbar'))
    scoliosis = float(request.args.get('scoliosis'))
    sacral = float(request.args.get('sacral'))
    thoracic = float(request.args.get('thoracic'))
    cervical = float(request.args.get('cervical'))

    return jsonify(low_back_pain.predict(pelvic, lumbar, scoliosis, sacral, thoracic, cervical))

if __name__ == '__main__':
    app.run(debug=True)
