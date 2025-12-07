# MLOps-Git-DVC-MLFlow-Dagshub
End-to-end Machine Learning pipeline using Git, DVC, MLFlow and Dagshub.

## Problem Statement
We have Pima Indians Diabetes Dataset which we can get from Kaggle. This dataset has features like Pregnancies, Glucose, BP, Insulin, Diabetes etc and "Outcome" is an output feature. So we need to design a model which takes this input features and it should be able to predict the outcome whether a specific person is Diabetic or not.</br>

A Repository has been created in DagsHub: https://dagshub.com/vr32288/mlpipeline.git

### Steps:
- First create a virtual environment, create requirements.txt which will have the following libraries included:
```bash
dvc
dagshub
scikit-learn
mlflow
dvc-s3
```

- Activate the virtual environment and then install libraries specified in requirements.txt
```bash
conda activate venv/
pip install -r requirements.txt
```
- Create a .gitignore file and put venv/ in that because we don't want our virtual env to be tracked.
- Since we need to start with data pre-processing, we need to have some kind of data first. So we'll upload the data intot he directory: data/raw/ & upload the .csv file. </br>
The outcome will be 0 or 1 based on the input features which will specify whether the person has diabetes or not. Based on this dataset we'll create a pipeline for data pre-processing, model training and then model evaluation.

- We'll create another folder "src" and files params.yml,__init__.py: through this file, we will be able to call this as a package, evaluate.py: specifically for evaluation, preprocess.py: for preprocessing such as reading data or any kind of feature engineering,train.py. - params.yml: will be used to setup some parameters. For preprocessing, we provide the input and then we save output for that preprocessed data. For training, we'll use the output that we saved from preprocessing as data, we provide path where the model will be saved.
<img width="371" height="265" alt="image" src="https://github.com/user-attachments/assets/e95e6baf-336a-4557-afa3-88435d0f21fe" /> </br>
In params.yml, we'll make calls to "preprocess" and "train", and each of have multiple parameters. "model" parameter is used to specify that after training the model, it will be saved in this path.

#### preprocess.py
In preprocess.py, import libraries required to preprocess like pandas, sys, import params.yml (in order to read params.yml), import os (so that we'll be able to set the path). </br>
Then we load parameters from params.yml (safe_load is used) and then we specify what key we need to call like here we have 2 keys: preprocess and train.</br>
Defined a function with 2 parameters: input_path and output_path. os.makedirs will be used to make directory where we store the output of preprocessor stage.
```bash
import pandas as pd
import sys
import yaml
import os

## Load paramters from params.yml
params=yaml.safe_load(open("params.yml"))['preprocess']

def preprocess(input_path,output_path):
    data=pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_csv(output_path,header=None,index=False)
    print(f"Preprocesses data saved to {output_path}")

if __name__=="__main__":
    preprocess(params["input"],params["output"])
```

#### train.py

- All the Tracking will be happen in the DgasHub repository. 
