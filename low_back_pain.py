import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# joblib is for model persistence so it doesn't have to train model at every request
from sklearn.externals import joblib
