import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Open csv file, and split into features and labels
# Note that for fall_risk (1 = Yes, 0 = No)
def predict_fall_risk(age, berg, gait):
    df = pd.read_csv('fall_risk.csv')
    features = df[['age','berg_balance','gait_speed']]
    labels = df[['fall_risk']]

    age_min = df[['age']].min()
    berg_balance_min = df[['berg_balance']].min()
    gait_speed_min = df[['gait_speed']].min()

    age_range = df[['age']].max() - age_min
    berg_balance_range = df[['berg_balance']].max() - berg_balance_min
    gait_speed_range = df[['gait_speed']].max() - gait_speed_min

    # Perform feature scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X = scaled_features
    y = np.ravel(labels)    # this changes our labels(column-vector) into a 1-D array

    # Perform 70:30 training:testing data split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    logistic = LogisticRegression()
    logistic.fit(X_train,y_train)

    # prediction is expecting data is a 1x3 two-dimensional array [[age, berg, gait]]
    # remember that the unknown data point must be scaled appropriately
        #scaling is done by: (x-minimum)/range

    # prediction = logistic.predict(X_test[0].reshape(1,-1))
    prediction = logistic.predict([[age, berg, gait]])
    score = logistic.score(X_test, y_test)

    if prediction == 0:
        print("Low fall risk")
    else:
        print("High fall risk")

    print("Model accuracy is:", score)

# TO INVOKE THIS FUNCTION:
predict_fall_risk(76,78,1.12)
