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
Defined a function with 2 parameters: input_path and output_path. os.makedirs will be used to make directory where we store the output of preprocessor stage. </br>
All these files preprocess.py, train.py etc will run as a pipeline as we connect all the steps.</br>
After running this file, a new folder "processed" will be created inside "data" folder. Inside that data.csv will be there which will have no column headers means the data is processesed now. Now we can do feature engineering or if we want to convert categorical features into numerical features or even change the scale of data.
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
First import libraries like pandas, randomforestclassifier, pickle (so that we create our pickle file), import yaml (so that we'll able to read parameters from params.yml) etc. </br>
A pickle file refers to a file containing a serialized Python object, typically a trained machine learning model, saved using the pickle module in Python. Pickling (Serialization is the process of converting a Python object (like a trained ML model, a dictionary, or a list) into a byte stream. This byte stream can then be saved to a file (the pickle file) or transmitted over a network. </br>
Since we're performing all the Logging and tracking in DgasHub, so we'll require the remote repository info related to logging experiments. We'll need to pass 3 environment variables: </br>
- MLFlow Tracking URI
- MLFlow tracking Username
- MLFlow tracking password </br>
These values we get from DagsHub UI, Click on Remote > Experiments, there we get the Tracking uri. Then Remote > Data > DVC and scroll to the bottom, there we get setup credentials. </br>

</br>
Create a Function for Hyperparamter tuning which first initializes a default RandomForestClassifier with no tuning yet. Sets up GridSearchCV, this tells scikit-learn to try different combinations of hyperparameters provided in param_grid: estimator=rf → the model you want to tune, param_grid=param_grid → dictionary of hyperparameters to try, cv=3 → 3-fold cross validation, n_jobs=-1 → uses all CPU cores (fastest) and verbose=2 → prints logs about progress while running. </br>
grid_search.fit(X_train, y_train): This step trains the model for each combination of hyperparameters, evaluates them using cross-validation and then finds the best combination.</br>
Then this function returns the entire GridSearchCV object. </br>
Train function gets the best model, predicts and evaluates and then prints Accuracy.
```bash
def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/vr32288/mlpipeline.mlflow")

    ## start the mlflow run
    with mlflow.start_run():
        # split the dataset into training and test set
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature=infer_signature(X_train,y_train)

        ## Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        ## Perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model=grid_search.best_estimator_

        ## predict and evaluate the model
        y_pred=best_model.predict(X_test)
        accuracy_score=accuracy_score(y_test,y_pred)
        print(f"Accuracy: {accuracy_score}")
```


- All the Tracking will be happen in the DagsHub repository. 
