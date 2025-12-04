## DVC (Data Version Control)
Manage and version images, audio, video and text files in storage and organize your ML modleing process into a reproducible workflow.

### Why DVC?
Even with all the success we've seen in machine learning, especially with deep learning and its applications in business, data scientists still lack best practices for organizing their projects and collaborating effectively. This is a critical challenge:
while ML algorithms and methods are no longer tribal knowledge, they are still difficult to develop, reuse, and manage.

### Use cases: 
- track and save data and machine learning models the same way you capture code
- create and switch between versions of data and ML models easily
- understand how datasets and ML artifacts were built in the first place
- compare model metrics among experiments
- adopt engineering tools and best practices in data science projects

Steps:
- First we will put the virtual env directory folder in .gitignore so that whenever we do "gitinit", this folder will not be tracked. </br>

- Then create requirements.txt and add "dvc" to it and then do
```bash
pip install -r requirements.txt
```
- Create a folder and create a file inside it that will be tracked using dvc.
- Then do "dvc init". And as soon as we do that, two new directories .dvc and .dvcignore will be created. Then when we do "git status", we'll see that these dvc directories and files will be automatically added, so we'll commit that.
- data/file1.txt if the file we need to track so we do
```bash
dvc add data/file1.txt
```
we'll get something like this:
<img width="1255" height="180" alt="image" src="https://github.com/user-attachments/assets/bb7ef313-2997-48a1-a13b-e3f7659963af" />
So this basically means that it is giving indication to Git that it does not to track "data" directory anymore as it is being tracked by dvc now. So we run:
```bash
git add data/.gitignore data/file1.txt.dvc
```

Also a new file "file1.txt.dvc" is created which will consist of hash-key which is mapped to the data we have i.e, data/file1.txt
<img width="797" height="301" alt="image" src="https://github.com/user-attachments/assets/3ea08f7e-bdff-4c2f-9d73-fb9b7236c87a" />


