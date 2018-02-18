from flask import Flask, jsonify, request
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn

app = Flask(__name__)

@app.route("/")
def hello():
    return "hello"

@app.route('/post', methods=['GET'])
def show_post():
    random = np.random.randint(1,100)
    post_id = request.args.get('id')
    post_name = request.args.get('name')
    output = {"id": post_id,
                "name": post_name,
                "random": random}
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
