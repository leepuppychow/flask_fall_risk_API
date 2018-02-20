# README

## Description

* This is my first experiment with the world of Python, Flask, and scikit-learn. The goal of this API is to classify patients as having high fall risk or low fall risk based on 3 features: age, Berg Balance Scale (percentage score), and gait speed (m/s).

* I am intending to call this API from my Parkinson's Health Tracker application.

* NOTE: the data used to train this logistic regression classifer model is fabricated. I would love to gather real data from physical therapists or existing studies if possible.

## Endpoints

To train the model the first time:

` GET request to: /fall_risk_api/v1/train_logistic `

To get a prediction (age in years, berg as percentage, gait in m/s):

` GET request to: /fall_risk_api/v1/predict?age=75&berg=88&gait=1.1`
