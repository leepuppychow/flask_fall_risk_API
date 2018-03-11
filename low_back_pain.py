import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib

def support_vector_machine():
    df = pd.read_csv('back_pain.csv')
    features = df[['pelvic_tilt','lumbar_lordosis_angle', 'scoliosis_slope', 'sacral_slope', 'thoracic_slope', 'cervical_tilt']]
    labels = df[['Class_att']]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    y = np.ravel(labels)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    model = SVC()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    data = {
        "model": model,
        "score": score
    }

    joblib.dump(data, "low_back_pain_SVM.pkl")

def predict(pelvic_tilt, lumbar_lordosis, scoliosis_slope, sacral_slope, thoracic_slope, cervical_tilt):
    data = joblib.load("./low_back_pain_SVM.pkl")

    # decided to hard-code the minimum and maximum values here:
    pelvic = (pelvic_tilt - (-6.555)) / (49.432 - (-6.555))
    lumbar = (lumbar_lordosis - 14.0) / (125.742 - 14.0)
    scoliosis = (scoliosis_slope - 7.008) / (44.341 - 7.008)
    sacral = (sacral_slope - 13.367) / (121.43 - 13.367)
    thoracic = (thoracic_slope - 7.038) / (19.324 - 7.038)
    cervical = (cervical_tilt - 7.031) / (16.821 - 7.031)

    prediction = data["model"].predict([[pelvic, lumbar, scoliosis, sacral, thoracic, cervical]]).item()

    if prediction == "Abnormal":
        return {"low_back_pain": "yes",
                "model_accuracy": data["score"]}
    else:
        return {"low_back_pain": "no",
                "model_accuracy": data["score"]}
