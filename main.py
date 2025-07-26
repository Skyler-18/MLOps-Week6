from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, predict_class

app = FastAPI()
clf = load_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "IRIS Classifier API"}

@app.post("/predict")
def predict(input: IrisInput):
    prediction = predict_class(clf, input)
    return {"prediction": prediction}
