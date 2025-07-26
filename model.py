from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

def predict_class(clf, input):
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = clf.predict(data)
    return int(prediction[0])
