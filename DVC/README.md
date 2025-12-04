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


- First we will put the virtual env directory folder in .gitignore so that whenever we do "gitinit", this folder will not be tracked. </br>

- Then create requirements.txt and add "dvc" to it and then do
```bash
pip install -r requirements.txt
```
- 
