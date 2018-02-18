import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Open csv file, and split into features and labels 
df = pd.read_csv('fall_risk.csv')
features = df[['age','berg_balance','gait_speed']]
labels = df[['fall_risk']]

# Perform feature scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

X = scaled_features
y = labels

# Perform 70:30 training:testing data split
X_train, X_test, y_train, t_test = train_test_split(X,y,test_size=0.3)
