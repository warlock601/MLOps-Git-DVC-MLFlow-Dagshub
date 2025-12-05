# MLOps-Git-DVC-MLFlow-Dagshub
End-to-end Machine Learning pipeline using Git, DVC, MLFlow &amp; Dagshub

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
- Since we need to start with data pre-processing, we need to have some kind of data first. So we'll upload the data intot he directory: data/raw/ & upload the .csv file.
