#!/usr/bin/env python
# coding: utf-8

# # Importing required Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


# Reading Training and Testing Files
train_files = glob.glob("D:\\pml-2022-smart\\train\\train\\*.csv")
test_files = glob.glob("D:\\pml-2022-smart\\test\\test\\*.csv")


# In[3]:


# Reading x train
x = []
for path in train_files:
    df = pd.read_csv(path, header=None).to_numpy().flatten()
    if len(df) == 450:
        x.append(df)
    elif len(df) < 450: 
        x.append(np.pad(df, pad_width=(0, 450-len(df))))
    else:
        x.append(np.delete(df,np.s_[450:]))

train_features = np.asarray(x)


# In[4]:


train_features.shape


# In[5]:


#Generating train labels
train_labels = pd.read_csv("D:\\pml-2022-smart\\train_labels.csv")['class'].to_numpy()


# In[6]:


# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, train_labels, test_size=2000, random_state=42,shuffle=False)


# ## SVM Model

# In[7]:


# Creat model object
svm = make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    SVC()
)


# In[8]:


# Get the parameters of the model
svm.get_params().keys()


# In[9]:


# define some parameters values
svm_params = {
    'pca__n_components': [30, 50],
    'svc__gamma':[0.01, 0.02, 0.03,'scale'],
    'svc__degree':[2,3,4,5],
    'svc__C':[1.0, 2.0]
}

# Initiating grid search object
svm_model = GridSearchCV(svm, svm_params, cv=5, n_jobs=1, verbose=10)
svm_model.fit(X_train, y_train)


# In[10]:


#Printing out the best score
svm_model.best_score_


# In[11]:


#Printing out the best parameters
svm_model.best_params_


# In[12]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(svm_model, X_val, y_val, cmap=plt.cm.Blues, ax=ax)
ax.set_title('Support vector Machine Confusion Matrix')


# In[13]:


#Printing out the validation score
svc_score = svm_model.score(X_val, y_val)
svc_score


# # KNeighbors Classifier Model

# In[14]:


# Creat model object
kn = make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    KNeighborsClassifier()
)

# Get the parameters of the model
kn.get_params().keys()


# In[15]:


# define some parameters values
kn_params = {
    'pca__n_components': [30, 50],
    'kneighborsclassifier__n_neighbors': [5, 6, 7],
    'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'kneighborsclassifier__p':[1, 2]
    }

# Initiating grid search object
kn_model = GridSearchCV(kn, kn_params, cv=5, n_jobs=1, verbose=10)
kn_model.fit(X_train, y_train)


# In[16]:


#Printing out the best score
kn_model.best_score_


# In[17]:


#Printing out the best parameters
kn_model.best_params_


# In[18]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(kn_model, X_val, y_val, cmap=plt.cm.Purples, ax=ax)
ax.set_title('K-Nearest Neighbors Confusion Matrix')


# In[19]:


#Printing out the validation score
kn_model.score(X_val, y_val)


# ## Random Forest Model

# In[20]:


# Creat model object
rf = make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    RandomForestClassifier(random_state=42)
)

# Get the parameters of the model
rf.get_params().keys()


# In[21]:


# define some parameters values
rf_params = {
    'pca__n_components': [30, 50],
    'randomforestclassifier__max_depth': [10, 20],
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__criterion':['gini', 'entropy']
    }

# Initiating grid search object
random_forest_model = GridSearchCV(rf, rf_params, cv=5, n_jobs=1, verbose=10)
random_forest_model.fit(X_train, y_train)


# In[22]:


#Printing out the best score
random_forest_model.best_score_


# In[23]:


#Printing out the best parameters
random_forest_model.best_params_


# In[24]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(random_forest_model, X_val, y_val, cmap=plt.cm.Greens, ax=ax)
ax.set_title('Random Forest Confusion Matrix')


# In[25]:


#Printing out the validation score
random_forest_model.score(X_val, y_val)


# ## XGBoost

# In[26]:


# Creat model object
xgboost = make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    xgb()
)

# Get the parameters of the model
xgboost.get_params().keys()


# In[27]:


# define some parameters values
xgboost_params = {
    'pca__n_components': [30, 50],
    'xgbclassifier__learning_rate': [0.3, 0.1, 0.2, 0.4],
    'xgbclassifier__n_estimators': [100, 200, 50, 70],
    'xgbclassifier__max_depth': [5, 6, 7, 9]
    }

# Initiating randomized search object
xgboost_model = RandomizedSearchCV(xgboost, xgboost_params,n_iter=10, cv=5, n_jobs=1, random_state=42, verbose=True)
xgboost_model.fit(X_train, y_train)


# In[28]:


#Printing out the best score
xgboost_model.best_score_


# In[29]:


#Printing out the best parameters
xgboost_model.best_params_


# In[30]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(xgboost_model, X_val, y_val, cmap=plt.cm.Greys, ax=ax)
ax.set_title('XG-Boost Confusion Matrix')


# In[31]:


#Printing out the validation score
xgboost_model.score(X_val, y_val)


# ## Bagging Classifier

# In[32]:


# Creat model object
bclf = make_pipeline(
    StandardScaler(),
    PCA(random_state=42),
    BaggingClassifier(random_state=42)
)

# Get the parameters of the model
bclf.get_params().keys()


# In[33]:


# define some parameters values
bclf_params = {
    'pca__n_components': [30, 50],
    'baggingclassifier__n_estimators': [10, 20, 30, 40, 50],
    }

# Initiating grid search object
bagging_classifier_model =GridSearchCV(bclf, bclf_params, cv=5, n_jobs=1, verbose=True)
bagging_classifier_model.fit(X_train, y_train)


# In[34]:


#Printing out the best score
bagging_classifier_model.best_score_


# In[35]:


#Printing out the best parameters
bagging_classifier_model.best_params_


# In[36]:


# Plotting the confution matrix
fig, ax = plt.subplots(figsize=(8,8))
plot_confusion_matrix(bagging_classifier_model, X_val, y_val, cmap=plt.cm.Reds, ax=ax)
ax.set_title('Bagging Classifier Confusion Matrix')


# In[37]:


#Printing out the validation score
bagging_classifier_model.score(X_val, y_val)


# ## Test

# In[38]:


#Loading test files
x = []
for path in test_files:
    df = pd.read_csv(path, header=None).to_numpy().flatten()


    if len(df) == 450:
        x.append(df)
    elif len(df) < 450: 
        x.append(np.pad(df, pad_width=(0, 450-len(df))))

    else:
        x.append(np.delete(df,np.s_[450:]))

X_test = np.asarray(x)


# In[39]:


X_test.shape


# In[40]:


#Making submission
import os
from os import listdir
 
# get the path/directory
folder_dir = "pml-2022-smart\\test\\test"
csv_ids = []
for csv in os.listdir(folder_dir):
        if csv.endswith(".csv"):
            csv_ids.append(csv[:5])

csv_ids = np.array(csv_ids)
csv_ids.shape
df = pd.DataFrame(csv_ids, columns = ['id'])


# In[46]:


# Make predictions
test_predictions = random_forest_model.predict(X_test)
test_predictions.shape


# In[47]:


test_predictions


# In[48]:


# concatinate the predictions and ids
final_df=pd.concat([df, pd.Series(test_predictions, name='class').reindex(df.index)], axis=1)


# In[49]:


final_df


# In[50]:


# save the predictions to csv file on the disk
final_df.to_csv('Submissions_rf.csv',index=False)


# In[ ]:




