import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import io
import graphviz
import seaborn as sns

#Uploading and Reading Dataset
dataset=pd.read_csv(r'C:\Users\KUCHAY_AASHIQ\Downloads\ckddata.csv')
print(dataset)

# Displaying Attributes
print(dataset.columns)

# Deleting unnecessary column(s)
dataset = dataset.drop(['id'],axis=1)
print(dataset.columns)

print(dataset.info())


# Handling attributes of object type
obj_cols=[col for col in dataset.columns if dataset[col].dtype=='object']
for col in obj_cols:
    print('-> {}\t:{}'.format(col,dataset[col].unique()))

dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')

obj_cols=[col for col in dataset.columns if dataset[col].dtype=='object']
for col in obj_cols:
    print('-> {}\t:{}'.format(col,dataset[col].unique()))


# Replacing non-numeric values with numeric values

dataset[['htn','pe','ane']]=dataset[['htn','pe','ane']].replace(to_replace={'yes':1,'no':0})
dataset[['dm']]=dataset[['dm']].replace(to_replace={'yes':1,'no':0,' yes':1,'\tno':0,'\tyes':1})
dataset[['cad']]= dataset[['cad']].replace(to_replace={'yes':1,'no':0,'\tno':0})
dataset[['rbc','pc']]=dataset[['rbc','pc']].replace(to_replace={'normal':1,'abnormal':0})
dataset[['pcc','ba']]=dataset[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
dataset[['appet']]=dataset[['appet']].replace(to_replace={'good':1,'poor':0})
dataset[['classification']]=dataset[['classification']].replace(to_replace={'ckd':1,'ckd\t':1,'notckd':0})
print(dataset)

print(dataset.dtypes)

# Count Mean Standard Deviation Minimum Value Maximum Value

print(dataset.describe().T)


# Plotting Histogram
pt=dataset.hist(figsize=(25,25))
print(pt)

# Count of Null Values
print(dataset.isnull().sum())
print(dataset.corr())

# Heatmap is de¦ned as a graphical representation of data using colors to visualize the value of the
# matrix.Used to represent more common values or higher activities brighter colors
plt.figure(figsize=(20,20))
matrix=np.triu(dataset.corr())
sns.heatmap(dataset.corr(), annot=True, cmap='Blues', mask=matrix)


# Handling Missing/Null Values
dataset.isnull().sum()
columns = dataset.columns.to_list()

# Replacing Missing/Null Values with mean values
dataset['classification'].fillna(value=0,inplace=True)
for i in (columns):
    dataset[i].fillna(dataset[i].mean(),inplace=True)
    
print(dataset.isnull().sum())

# Feature Selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

ind_col=[col for col in dataset.columns if col!='classification']
dep_col='classification'

x=dataset[ind_col]
y=dataset[dep_col]

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(x,y)
datascores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(x.columns)
features_rank=pd.concat([dfcolumns,datascores],axis=1)
features_rank.columns=['Features','Score']


# Fetch largest 10 values of Score column

selected_columns=features_rank.nlargest(10,'Score')['Features'].values
X_new=dataset[selected_columns]
dataset = pd.concat([X_new, y], axis=1)
selected_columns
print(dataset)

# Histogram
pt=dataset.hist(figsize=(15,15))
print(pt)

# Heatmap
plt.figure(figsize=(10,10))
matrix=np.triu(dataset.corr())
sns.heatmap(dataset.corr(),annot=True,cmap='Blues', mask=matrix)

# Shuffling the data
from sklearn.utils import shuffle
dataset=shuffle(dataset)
print(dataset)

# Preparing data for modelling
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Test-Train Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)
print(x_train.shape,x_test.shape)


# Decision Tree Classi¦er

dtree=DecisionTreeClassifier(max_depth=7,criterion='entropy')
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print("\n\n\t DECISION TREE CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, y_pred))
print("classification report is:\n",classification_report(y_test,y_pred))
print("Confusion matrix:\n",confusion_matrix(y_test,y_pred))

# KNN Classi¦er
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
pred2=knn.predict(x_test)
print("\n\n\t K NEAREST NEIGHBOR CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred2))
print("classification report is::\n",classification_report(y_test,pred2))
print("confusion matrix:\n",confusion_matrix(y_test,pred2))

# SVM Classi¦er
from sklearn.svm import SVC
svc_model=SVC(C=70)
svc_model.fit(x_train,y_train)
pred3=svc_model.predict(x_test)
print("\n\n\t SUPPORT VECTOR MACHINE CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred3))
print("classification report is::\n",classification_report(y_test,pred3))
print("confusion matrix:\n",confusion_matrix(y_test,pred3))


# RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=100,criterion='entropy')
rf.fit(x_train,y_train)
pred4= rf.predict(x_test)
print("\n\n\t RANDOM FOREST CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred4))
print("classification report is::\n",classification_report(y_test,pred4))
print("confusion matrix:\n",confusion_matrix(y_test,pred4))

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear',multi_class='ovr')
logreg.fit(x_train, y_train)
pred5 = logreg.predict(x_test)
print("\n\n\t LOGISTIC REGRESSION CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred5))
print("classification report is::\n",classification_report(y_test,pred5))
print("confusion matrix:\n",confusion_matrix(y_test,pred5))

# GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
pred6 = gaussian.predict(x_test)
print("\n\n\t GAUSSIAN NAIVE BAYES CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred6))
print("classification report is::\n",classification_report(y_test,pred6))
print("confusion matrix:\n",confusion_matrix(y_test,pred6))

# Ada Boost Classi¦er
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(x_train, y_train)
pred7 = ada.predict(x_test)
print("\n\n\t ADA BOOST CLASSIFIER\n")
print("\nAccuracy Score :", accuracy_score(y_test, pred7))
print("classification report is::\n",classification_report(y_test,pred7))
print("confusion matrix:\n",confusion_matrix(y_test,pred7))

# CONCLUSION

dcc=accuracy_score(y_test,y_pred)
knnc=accuracy_score(y_test,pred2)
svmc=accuracy_score(y_test,pred3)
rfc=accuracy_score(y_test,pred4)
lgr= accuracy_score(y_test, pred5)
gnb= accuracy_score(y_test, pred6)
abc=accuracy_score(y_test, pred7)

models=['Support Vector Classifier','K-Nearest Neighbour Classifier','Decision Tree Classifier', 'Logistic Regression','Gaussian Naive Bayes', 'Random ForestClassifier', 'Ada Boot'] 
scores=[svmc,knnc,dcc, lgr, gnb, rfc, abc]

plt.figure(figsize=(10,5))
sns.barplot(x=scores, y=models)
plt.show()

score_table=pd.DataFrame({'model':models,'score':scores})
score_table.sort_values(by='score',axis=0,ascending=True)

print(score_table)

# Saving the Model

import pickle
saved_model = pickle.dumps(ada)
ada_from_pickle = pickle.loads(saved_model)
print(ada_from_pickle.predict(x_test))

# Testing The Model

# Patient No. 238
pre = ada.predict([[8406.122449, 201.0, 241.000000, 13.400000, 28.000000, 1.016949, 9.400000, 72.0, 0.450142, 1.0]])
if pre == 1:
    print("Patient has CKD")
else:
    print("Patient does not have CKD")
# Patient No. 70
pre = ada.predict([[8300.000000, 360.0, 19.000000, 0.700000, 44.000000, 0.000000, 15.200000, 61.0, 4.0, 1.0]])
if pre == 1:
    print("Patient has CKD")
else:
    print("Patient does not have CKD")
#Patient No. 362
pre = ada.predict([[10300.000000, 89.0, 19.000000, 1.100000, 40.000000, 0.000000, 15.000000, 33.0, 0.000000, 0.0]])
if pre == 1:
    print("Patient has CKD")
else:
    print("Patient does not have CKD")
