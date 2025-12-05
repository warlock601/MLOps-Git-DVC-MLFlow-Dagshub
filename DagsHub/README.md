# DagsHub
Remote repository used to track data with Git and DVC. Dagshub integrates easily with MLOps stacks like Kaggle, Pandas, Github, aws, kubeflow, pytorch and many more.</br>


### Use cases:
- Dataset Management and Versioning
- Experiment Tracking
- Annotating data
- Model Registry & Deployment

### Difference b/w GitHub and DagsHub
With Github we can only maintain the versioning of our code but with Dagshub we can upload data to Dagshub (DVC), track code and Log experiments with MLFlow. We can run MLFlow server over a Dagshub account.

Link for Dagshub repo:
https://dagshub.com/vr32288/demoagshub.git

Steps:
- First we create a virtual environment and then activate it.
```bash
conda create -p venv python==3.10
```
- Then we create requirements.txt and .gitignore files. In the .gitignore file, we will add the virtual environment directory so that it is not tracked.
- In requirements.txt add these 2 libraries
```bash
dagshub
dvc
```
- Activate the virtual env
```bash
conda activate venv/
```
- Add a data file like a csv file in this directory: data/data.csv. Our main aim is to track this entire versioning of the data over here and we also need to make sure that we push all this versioning info inside our DagsHub repository which we have created.
- We want to track data/data.csv so we'll use commands:
```bash
dvc init
dvc add data/data.csv
```
- We'll also need these: data/data.csv.dvc data/.gitignore to be tracked by git itself
```bash
git add data/data.csv.dvc data/.gitignore
git commit -m "Added data.csv with DVC"
```
- Now we will push everything into DagsHub repository. In Dagshub, Click on Remote > Data > DVC. We need to setup DagsHub DVC remote where we store all our setup. We need to configure it.
```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/vr32288/demodagshub.s3
```
- Also setup security credentials. Both this and the above step are there in the Data > DVC option in the Dagshub repository.
```bash
dvc remote modify origin --local access_key_id 13ba0c3ed37e7f905ccc89c80efa51057ffaed5e
dvc remote modify origin --local secret_access_key 13ba0c3ed37e7f905ccc89c80efa51057ffaed5e
```
We can check the remote list to see if s3-origin is created or not.
```bash
dvc remote list
```
- After that we do pull so that whatever recent info is there. But we'l get an error: dvc-s3 library required. So we put it in requirements.txt and install it.
```bash
dvc pull -r origin
```
- Push to DVC
```bash
dvc push -r origin
```
- Then we need to add, commit & push using Git and we'see all the data in DagsHub repo.
<img width="1130" height="371" alt="image" src="https://github.com/user-attachments/assets/56e09d19-f8a9-4579-a084-276489cb476d" />
<img width="1428" height="771" alt="image" src="https://github.com/user-attachments/assets/c97fb9c7-a506-4482-9b55-ab4e93192b0c" />

- What happens when we perform any changes in the data? Then we do dvc add, git add, git commit, dvc pull -r origin, dvc push -r origin, git push origin main. This way we'll be able to track the changes.


