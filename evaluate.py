import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/vr32288/mlpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="vr32288"
os.environ['MLFLOW_TRACKING_PASSWORD']="13ba0c3ed37e7f905ccc89c80efa51057ffaed5e"

## Load the parameters from params.yml
params=yaml.safe_load(open("params.yml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/vr32288/mlpipeline.mlflow")


    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    ## log metrics to mlflow
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy}")


if __name__=="__main__":
    evaluate(params["data"],params["model"])
