
# Titanic - Machine Learning from Disaster

![App Screenshot](https://github.com/bharathngowda/machine_learning_heart_diesease_prediction/blob/main/dataset-cover.jpg)

### Table of Contents

1. [Problem Statement](#Problem-Statement)
2. [Data Pre-Processing](#Data-Pre-Processing)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Model Training](#Model-Building)
5. [Model Selection](#Model-Selection)
6. [Model Evaluation](#Model-Evaluation)
7. [Dependencies](#Dependencies)
8. [Installation](#Installation)

### Problem Statement

This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
this date. The "target" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. The goal is identify whether a patient has heart disease or not based on other features.

**Quick Start:** [View](https://github.com/bharathngowda/machine_learning_heart_diesease_prediction/blob/main/Heart%20DiseasePrediction.ipynb) a static version of the notebook in the comfort of your own web browser

### Data Pre-Processing

- Loaded the train and test data
- Checking if the data is balanced i.e. whether the count of survived and not survived is equal or not in train set.
- Splitting the dataset to test and train set as there is no separate test set.
- Checkig for null values


### Exploratory Data Analysis

- **Correlation Plot** to determine if there is a linear relationship between the 2 variables and the strength of the relationship


### Model Training

Models used for the training the dataset are - 

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [Linear Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
- [Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


### Model Selection

Since the dataset is imbalanced, I have used [f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) as my scorer and used [k-fold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
to select the model with highest **'f1 score'**.

### Model Evaluation

I fit the final model on the train data and predicted the survival status for the test data and obtained the below result-

| Metric        | Score    |
| :--------     | :------- |
| F1 Score	    |0.811881  |
| Precision	    |0.803922  |
| Recall	    |0.820000  |
| Accuracy	    |0.791209  |
| ROC	        |0.788049  |
| Cohens Kappa	|0.577365  |

Confusion Matrix and ROC Curve

![App Screenshot](https://github.com/bharathngowda/machine_learning_heart_diesease_prediction/blob/main/ROC%20%26%20CM.PNG)

### Dependencies
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

### Installation

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/bharathngowda/machine_learning_heart_diesease_prediction/archive/refs/heads/main.zip) or execute this from the terminal:
`git clone https://github.com/bharathngowda/machine_learning_heart_diesease_prediction.git`

2. Install [virtualenv](http://virtualenv.readthedocs.org/en/latest/installation.html).
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with `virtualenv env`.
4. Activate the environment with `source env/bin/activate`
5. Install the required dependencies with `pip install -r requirements.txt`.
6. Execute `ipython notebook` from the command line or terminal.
7. Click on `Heart DiseasePrediction.ipynb` on the IPython Notebook dasboard and enjoy!
8. When you're done deactivate the virtual environment with `deactivate`.
