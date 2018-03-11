# README

## Description

* This is my first experiment with the world of Python, Flask, and scikit-learn. The goal of this API is to classify patients as having high fall risk or low fall risk based on 3 features: age, Berg Balance Scale (percentage score), and gait speed (m/s). I ended up using a Logistic Regression Binary Classification model.

* This API can also classify patient as likely to have or not have low back pain based on 6 features: pelvic tilt, lumbar lordosis angle, thoracic slope, cervical tilt, scoliosis slope, and sacral slope. For this, I used a Support Vector Machine for this binary classification task.

* NOTE: the data used to train this logistic regression classifer model IS FABRICATED. I would love to gather real data from physical therapists or existing studies if possible.

* NOTE: data for the low back pain model was obtained from the source below. I was unable to find the authors of this dataset in order to credit them.

https://www.kaggle.com/alihussain1993/lower-back-pain-symptoms-datasetlabelled/data


## Endpoints

Fall Risk Prediction (age in years, berg as percentage, gait in m/s):

`Ex: GET request to: https://fall-risk-api.herokuapp.com/v1/predictfallrisk?age=75&berg=88&gait=1.1`

Low Back Pain Prediction:

`Ex: GET request to: http://fall-risk-api.herokuapp.com/v1/predictlowbackpain?pelvic=25&lumbar=30&scoliosis=40&sacral=100&thoracic=15&cervical=15`
