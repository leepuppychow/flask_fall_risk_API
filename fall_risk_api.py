from flask import Flask, jsonify, request
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route("/")
def hello():
    return "Fall risk prediction API (logistic regression multivariate classifier)"

@app.route('/predict', methods=['GET'])
def predict():
    age = float(request.args.get('age'))
    berg = float(request.args.get('berg'))
    gait = float(request.args.get('gait'))

    def predict_fall_risk(age, berg, gait):
        df = pd.read_csv('fall_risk.csv')
        features = df[['age','berg_balance','gait_speed']]
        labels = df[['fall_risk']]

        age_min = df[['age']].min()
        berg_min = df[['berg_balance']].min()
        gait_min = df[['gait_speed']].min()

        age_range = df[['age']].max() - age_min
        berg_range = df[['berg_balance']].max() - berg_min
        gait_range = df[['gait_speed']].max() - gait_min

        # Perform feature scaling
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        X = scaled_features
        y = np.ravel(labels)    # this changes our labels(column-vector) into a 1-D array

        # Perform 70:30 training:testing data split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

        logistic = LogisticRegression()
        logistic.fit(X_train,y_train)

        # Scale the input values: (value-min)/range
        # Note .item() returns the first element of the data structure
        # age_scaled is a pandas Series object, which has the value AND the datatype
        # We only want the value.
        age_scaled = ((age - age_min) / age_range).item()
        berg_scaled = ((berg - berg_min) / berg_range).item()
        gait_scaled = ((gait - gait_min) / gait_range).item()

        prediction = logistic.predict([[age_scaled, berg_scaled, gait_scaled]])
        score = logistic.score(X_test, y_test)

        if prediction == 0:
            return {"Fall Risk": "LOW",
                    "Model Accuracy": score}
        else:
            return {"Fall Risk": "HIGH",
                    "Model Accuracy": score}


    return jsonify(predict_fall_risk(age, berg, gait))




if __name__ == '__main__':
    app.run(debug=True)
