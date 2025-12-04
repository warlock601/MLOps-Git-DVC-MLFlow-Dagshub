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
- First we will put the virtual env directory folder in .gitignore so that whenever we do "git init", this folder will not be tracked. </br>

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
So this basically means that it is giving indication to Git that it should not track "data" directory anymore as it is being tracked by dvc now. So we run:
```bash
git add data/file1.txt.dvc
git add data/.gitignore 
```

Also a new file "file1.txt.dvc" is created which will consist of hash-key which is mapped to the data we have i.e, data/file1.txt. If we change this data and add some other data in data/file1.txt, then again this hash value will change. This mapping is done inside the .dvc/cache/ there will be that hash key.
<img width="797" height="301" alt="image" src="https://github.com/user-attachments/assets/3ea08f7e-bdff-4c2f-9d73-fb9b7236c87a" />

- When any changes will be made to this file data/file1.txt and we do "dvc add data/file1.txt", the hash value in file1.txt.dvc will change.
<img width="803" height="183" alt="image" src="https://github.com/user-attachments/assets/1877a698-bd4f-4593-8c7f-3e6176974ed9" />
<img width="1020" height="375" alt="image" src="https://github.com/user-attachments/assets/6e4cc249-a59b-4ebf-9925-b3e2dd03f15d" />

- Inside cache directory as well, we will have one more directory created precursor to the already existing one. The new one will have the neer version of file while the oolder one will have the previous version of file.
<img width="886" height="202" alt="image" src="https://github.com/user-attachments/assets/1d4204f3-0cae-41d5-a2af-37c198b4ef92" />
<img width="887" height="216" alt="image" src="https://github.com/user-attachments/assets/2d93b3f3-4d15-424e-8cbd-a50240b65fd0" />

- After this we do "git commit" so that all of these changes are logged.

- If we want to switch to a different version of data, first switch to a different branch (or create one and then switch) and then do "git log", it will show the list of all the commits and from there we can choose the commit we want to backtrack to. Like if we want to see a particular version of data, we get the list of commits by "git log" and then "get checkout commit-hash". We'll see that file1.txt.dvc will get changed and get the hash-key of that commit.

- But the file1.txt will not change. For it to change as per the commit-hash we checked out to, we need to do "dvc checkout". And then we'll see that file1.txt has also been restored to the previous/selected version.

- Now if we want to switch to the latest version, "git checkout master" (That's the reason why we switched to a different branch). Then do "dvc checkout"
```bash
git checkout master
dvc checkout
```
