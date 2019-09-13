#!/usr/bin/env python
# coding: utf-8

# In[56]:


from pandas import read_csv
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

Titanic_DataSet_org= read_csv("D:/UPitt/Studies/DA/Project/all/train.csv")
print(Titanic_DataSet_org.isnull().sum())
#print(Titanic_DataSet.loc[:, Titanic_DataSet.isna().any()])
#Titanic_DataSet_mod= Titanic_DataSet_org[Titanic_DataSet_org['Age'].isnull()==False]
Titanic_DataSet_org['Age'].fillna(Titanic_DataSet_org['Age'].mean(), inplace = True)
#Titanic_DataSet_mod= Titanic_DataSet_mod[Titanic_DataSet_mod['Embarked'].isnull()==False]
#Titanic_DataSet_mod = Titanic_DataSet_mod.reset_index(drop=True)
# print("T1:", Titanic_DataSet_mod)

# print("Cols", list(Titanic_DataSet_mod))
labelencoder = LabelEncoder()
# ohe = OneHotEncoder()
Titanic_DataSet_org['Sex'] = labelencoder.fit_transform(Titanic_DataSet_org['Sex'])

# embarked = Titanic_DataSet_mod['Embarked']

# embarked = labelencoder.fit_transform(embarked)

# embarked_enc = ohe.fit_transform(embarked.reshape(-1,1)).toarray()

# embarked_enc = pd.DataFrame(embarked_enc, columns=labelencoder.classes_)
# print("11",embarked_enc.shape)
# frames = [Titanic_DataSet_mod, embarked_enc]
# print("T2:", Titanic_DataSet_mod)
# Titanic_DataSet_mod=pd.concat(frames, axis=1)

# Titanic_DataSet_moda= Titanic_DataSet_mod[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare']]
# Titanic_DataSet_moda
# # Titanic_DataSet_moda['C']= embarked_enc.iloc[:, 0]
# # Titanic_DataSet_moda['Q']= embarked_enc.iloc[:, 1]
# # Titanic_DataSet_moda['S']= embarked_enc.iloc[:, 2]

# print(embarked_enc)
# print(Titanic_DataSet_mod)

#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
X= Titanic_DataSet_org[['Pclass', 'Sex', 'Age', 'SibSp', 'PassengerId','Parch','Fare']].values
# arr= numpy.isnan(X)
# arr1= arr[arr==False]
# print(arr1)
Y= Titanic_DataSet_org.Survived
# X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state= 0)
# scaler=preprocessing.StandardScaler().fit(X_train)
# X_train_transformed=scaler.transform(X_train) 
# X_test_transformed=scaler.transform(X_test) 

# RidgeModel = Ridge(alpha=50).fit(X_train_transformed, Y_train)
# print(RidgeModel.score(X_test_transformed, Y_test))

# LDAmodelFitted = LinearDiscriminantAnalysis().fit(X_train_transformed, Y_train)
# print("Accuracy of the LDA model is: ", LDAmodelFitted.score(X_test_transformed, Y_test))

# QDAmodelFitted = QuadraticDiscriminantAnalysis().fit(X_train_transformed, Y_train)
# print("Accuracy of the QDA model is: ", QDAmodelFitted.score(X_test_transformed, Y_test))

# LogRegModel= LogisticRegression(C=10).fit(X_train_transformed, Y_train)
# print("R-squared metric of the logistic regression model is: ", LogRegModel.score(X_test_transformed, Y_test))

# knn= KNeighborsClassifier(n_neighbors = 5)
# knn.fit(X_train_transformed, Y_train)
# print("R-squared metric of KNN model is: ", knn.score(X_test_transformed, Y_test))

import warnings
warnings.filterwarnings('ignore')

X_trainval, X_test, Y_trainval, Y_test= train_test_split(X, Y, random_state= 0)
# arr=np.isnan(X_trainval).any()
# print(arr)
scaler=preprocessing.StandardScaler().fit(X_trainval)
X_trainval_transformed=scaler.transform(X_trainval) 
X_test_transformed=scaler.transform(X_test) 
num_folds= 20

best_score=0
for val in [0.01, 0.1, 1, 2, 10, 100]:
    LogRegModel= LogisticRegression(C=val)
    scores= cross_val_score(LogRegModel, X_trainval_transformed, Y_trainval, cv=num_folds)
    score= np.mean(scores)
    if(score > best_score):
        best_score= score
        best_parameter= val

FinalLogRegModel= LogisticRegression(C=best_parameter).fit(X_trainval_transformed, Y_trainval) 
print("The best tuning parameter for regularization in this dataset is: ",best_parameter)
print("R-squared metric of the logistic regression model after 5-fold cross validation when tuning parameter = ",best_parameter,
      "is:", FinalLogRegModel.score(X_test_transformed, Y_test))

# import statsmodels.formula.api as sm
 
# model = sm.Logit(Y_trainval, X_trainval_transformed)
 
# result = model.fit()
# print(result.summary())

Titanic_DataSet_test = read_csv("D:/UPitt/Studies/DA/Project/all/test.csv")
print(list(Titanic_DataSet_test))
Titanic_DataSet_test['Sex'] = labelencoder.fit_transform(Titanic_DataSet_test['Sex'])
Titanic_DataSet_test['Age'].fillna(Titanic_DataSet_test['Age'].mean(), inplace = True)
Titanic_DataSet_test['Fare'].fillna(Titanic_DataSet_test['Fare'].mean(), inplace = True)
#display(Titanic_DataSet_test[['Pclass', 'Sex', 'Age', 'SibSp', 'PassengerId']])
X_test= Titanic_DataSet_test[['Pclass', 'Sex', 'Age', 'SibSp', 'PassengerId','Parch','Fare']]
print(FinalLogRegModel.predict(X_test_transformed))
print(X_test.isnull().sum())
#Titanic_DataSet_test= Titanic_DataSet_test[Titanic_DataSet_test['Age'].isnull()==False]
prediction_test= FinalLogRegModel.predict(X_test.values)
print(prediction_test)
result= pd.concat([X_test.PassengerId, pd.Series(prediction_test.reshape(len(prediction_test)))], axis=1)
# result.reset_index(inplace=True) # Resets the index, makes factor a column
# result.drop("Factor",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
result.rename(columns={'PassengerId': 'PassengerId', 0: 'Survived'}, inplace=True)
result.to_csv("submission_logreg.csv", index=False)

# from sklearn.model_selection import GridSearchCV

# # create param grid object 
# forrest_params = dict(     
#     max_depth = [n for n in range(1, 14)],     
#     min_samples_split = [n for n in range(2, 11)], 
#     min_samples_leaf = [n for n in range(1, 5)],     
#     n_estimators = [n for n in range(1, 60, 10)],
# )

# forrest_params2 = dict(     
#     max_depth = [n for n in range(1, 102, 1)],     
#     min_samples_split = [n for n in range(2, 102, 1)], 
#     min_samples_leaf = [n for n in range(1, 102, 1)],     
#     n_estimators = [n for n in range(1, 102, 1)],
# )

# from sklearn.ensemble import RandomForestClassifier
# forestModel= RandomForestClassifier () 
# forest_cv = GridSearchCV(estimator=forestModel, param_grid=forrest_params2, cv=5) 
# forest_cv= forest_cv.fit(X_trainval_transformed, Y_trainval)
# print("Forest", forest_cv.score(X_test_transformed, Y_test))

# from sklearn.svm import SVC

# C = [n for n in range(1, 102, 1)]
# Gamma = [n for n in range(1, 102, 1)]

# num_folds= 5
# best_score=0
# for val in C:
#     for gam in Gamma:
#         SvmModel= SVC(kernel='rbf', gamma=gam, C=val)
#         scores= cross_val_score(SvmModel, X_trainval_transformed, Y_trainval, cv=num_folds)
#         score= np.mean(scores)
#         if(score > best_score):
#             best_score= score
#             best_regparameter= val
#             best_gamma=gam

# FinalSvmModel= SVC(kernel='rbf', gamma=best_gamma, C=best_regparameter).fit(X_trainval_transformed, Y_trainval)
# print("The best tuning parameter for regularization in this dataset is: ", best_regparameter)
# print("The best RBF parameter Gamma value in this dataset is: ", best_gamma)

# print("R-squared metric of the SVM model after 5-fold cross validation when tuning parameter = ",best_regparameter,
#       "and gamma = ",best_gamma,"is:", FinalSvmModel.score(X_test_transformed, Y_test))



from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
random_forest.fit(X_trainval_transformed, Y_trainval)
print(X_test.values)
Y_pred_rf = random_forest.predict(X_test.values)
#random_forest.score(X_trainval_transformed, Y_trainval)
acc_random_forest = round(random_forest.score(X_test_transformed, Y_test) * 100, 2)
print(acc_random_forest)
print(Y_pred_rf)


# In[ ]:




