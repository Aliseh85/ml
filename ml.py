# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

path = "/content/drive/MyDrive/Churn_Modelling.csv"
df = pd.read_csv(path)
X1=df.drop(['Exited'],axis=1)
Y1=df['Exited']
def add_missing_values(X_full):
             
             Col_names=X_full.columns
             X_full=X_full.to_numpy()
             rng = np.random.RandomState(4)
             n_samples, n_features = X_full.shape
        
             # Add missing values in 75% of the lines
             missing_rate = 0.75
             n_missing_samples = int(n_samples * missing_rate)
        
             missing_samples = np.zeros(n_samples, dtype=bool)
             missing_samples[:n_missing_samples] = True
        
             rng.shuffle(missing_samples)
             missing_features = rng.randint(0, n_features, n_missing_samples)
             X_missing = X_full.copy()
             X_missing[missing_samples, missing_features] = np.nan
             X_missing=pd.DataFrame(X_missing)
             X_missing.columns=Col_names
             return X_missing

X_missing=add_missing_values(X1)

#make copy to work with check and drop irrelvantcolumns
X_missing.drop(['RowNumber','Surname'],inplace=True,axis=1)
X_mean=X_missing.copy(deep=True)
X_reg=X_missing.copy(deep=True)

#fill missing data with regression
def random_imputation(X_reg, feature):
    number_missing = X_reg[feature].isnull().sum()
    observed_values = X_reg.loc[X_reg[feature].notnull(), feature]
    X_reg.loc[X_reg[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    return X_reg
    
missing_columns = list(X_missing.columns)
for feature in missing_columns:
    X_reg[feature + '_imp'] = X_reg[feature]
    X_reg = random_imputation(X_reg, feature)

for feature in missing_columns:
  X_reg=X_reg.drop([feature],axis=1)
print(X_reg)

#convert object type dataframe to numeric
X_reg['IsActiveMember_imp'].astype(str).astype(int)
X_reg['EstimatedSalary_imp'].astype(str).astype(float)
X_reg['Tenure_imp'].astype(str).astype(int)
X_reg['Balance_imp'].astype(str).astype(float)
X_reg['NumOfProducts_imp'].astype(str).astype(int)
X_reg['HasCrCard_imp'].astype(str).astype(int)
X_reg['NumOfProducts_imp'].astype(str).astype(int)
X_reg['Gender_imp'].astype(str)
X_reg['Geography_imp'].astype(str)
X_reg['CreditScore_imp'].astype(str).astype(int)
X_reg['CustomerId_imp'].astype(str).astype(int)
X_reg['Age_imp'].astype(str).astype(int)

#c mean way to fill misssing values
X_mean['Geography'].fillna(X_mean['Geography'].mode()[0], inplace=True)
X_mean['Gender'].fillna(X_mean['Gender'].mode()[0], inplace=True)
X_mean['Age'].fillna(value=X_mean['Age'].mode()[0], inplace=True)
X_mean['CreditScore'].fillna(value=X_mean['CreditScore'].median(), inplace=True)
X_mean['Tenure'].fillna(value=X_mean['Tenure'].median(), inplace=True)
X_mean['Balance'].fillna(value=X_mean['Balance'].mean(), inplace=True)
X_mean['NumOfProducts'].fillna(value=X_mean['NumOfProducts'].mean(), inplace=True)
X_mean['HasCrCard'].fillna(value=X_mean['HasCrCard'].median(), inplace=True)
X_mean['IsActiveMember'].fillna(value=X_mean['IsActiveMember'].median(), inplace=True)
X_mean['EstimatedSalary'].fillna(value=X_mean['EstimatedSalary'].mean(), inplace=True)
X_mean.dropna(inplace=True)
print(X_mean.isnull().sum())
print(X_mean.info())
print(X_mean.describe())
print(X_mean.head())

#extra check for misssing value just to validate
print(X_mean.isnull().sum())
print(X_reg.isnull().sum())
print(X_reg)

#get rid of categrical columns 
le = LabelEncoder()
le.fit(X_mean['Geography'])
X_mean['new_Geography'] = le.transform(X_mean['Geography'])
le.fit(X_mean['Gender'])
X_mean['new_Gender'] = le.transform(X_mean['Gender'])
le.fit(X_mean['CustomerId'])
X_mean['new_CustomerId'] = le.transform(X_mean['CustomerId'])
X_mean.drop(['Gender','CustomerId','Geography'],inplace=True,axis=1)
print(X_mean.head())
###################
le = LabelEncoder()
le.fit(X_reg['Geography_imp'])
X_reg['new_Geography'] = le.transform(X_reg['Geography_imp'])
le.fit(X_reg['Gender_imp'])
X_reg['new_Gender'] = le.transform(X_reg['Gender_imp'])
le.fit(X_reg['CustomerId_imp'])
X_reg['new_CustomerId'] = le.transform(X_reg['CustomerId_imp'])
X_reg.drop(['Gender_imp','CustomerId_imp','Geography_imp'],inplace=True,axis=1)
print(X_reg.head())

X_reg.dropna(inplace=True)
X_reg1=X_reg

#normalize
transformer = Normalizer().fit(X_reg)
Normalizer()
X_reg=transformer.transform(X_reg)
X_reg = pd.DataFrame(X_reg, columns=X_missing.columns)
print(X_reg)
transformer1 = Normalizer().fit(X_mean)
Normalizer()
X_mean=transformer1.transform(X_mean)
X_mean = pd.DataFrame(X_mean, columns=X_missing.columns)
print(X_mean)

Y1.value_counts().plot(kind='bar')
print(Y1.value_counts())
#our data is imbalnced by 8 to 1 for that we need to balance the data in a propper way for more realstic results 
class_count_0, class_count_1 = Y1.value_counts()
# Separate class
class_0 = Y1[Y1 == 0]
class_1 = Y1[Y1 == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

# class count
class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1], axis=0)
class_1_over = class_1.sample(class_count_0, replace=True)
test_over = pd.concat([class_1_over, class_0], axis=0)
# plot the count after over-sampeling
test_over.value_counts().plot(kind='bar', title='count (target)')
class_count_0, class_count_1 = test_over.value_counts()
class_0 = test_over[test_over == 0]
class_1 = test_over[test_over == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

"""we checked if there is aneed for under sampling the count of lost data in such way and decides to do after all an over sampling so wo wont lose so much data and
and dont make a big inflection to the data and results by adding the missing count of data to the missing targer so it become equal to other side
"""

#split data to train and test with using the balanced data we made before and the 2 tables
dd=pd.DataFrame(test_over)
n = 10000
y=dd.iloc[:n]
n1 = 9441
y1=dd.iloc[:n1]
X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.20, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_mean, y1, test_size=0.20, random_state=42)

#fist model catboost
cbc = CatBoostClassifier()
param={'depth':[4,5,6,7,8,9,10],
      'learning_rate':[0.01,0.02,0.03,0.04],
      'iterations':[10,20,30,40,50,60,70,80,90,100]}
Grid_cbc=GridSearchCV(estimator=cbc,param_grid=param,cv=2,n_jobs=-1)
Grid_cbc.fit(X_train,y_train)
print(" the best estimator:",Grid_cbc.best_estimator_)
print("\n the best estimator:",Grid_cbc.best_score_)
print("the best estimator:",Grid_cbc.best_params_)
cbc1=CatBoostClassifier
Grid_cbc1=GridSearchCV(estimator=cbc,param_grid=param,cv=2,n_jobs=-1)
Grid_cbc1.fit(X_train1,y_train1)
print(" the best estimator:",Grid_cbc1.best_estimator_)
print("\n the best estimator:",Grid_cbc1.best_score_)
print("the best estimator:",Grid_cbc1.best_params_)

#second model logistic reggression
c_space = np.logspace(-5,8,15)
param_grid={'C':c_space}
logreg = LogisticRegression()
# fit the model with data
logreg_cv=GridSearchCV(logreg,param_grid,cv=5)
logreg.fit(X_train, y_train)
logreg_cv.fit(X_train,y_train)
print("tuned logistic para:{}".format(logreg_cv.best_params_))
print("best score is:{}".format(logreg_cv.best_score_))
c_space=np.logspace(-5,8,15)
logreg_cv1=GridSearchCV(logreg,param_grid,cv=5)
logreg.fit(X_train1, y_train1)
logreg_cv1.fit(X_train1,y_train1)
print("tuned logistic para:{}".format(logreg_cv1.best_params_))
print("best score is:{}".format(logreg_cv1.best_score_))

"""d) they gave us the abillity to get the best action and results we may need 
from the model with tuning with gridsearch method that control the learning proccess of the machine
 as they remain constant during all the work till giving best result as we needed

"""

# qustion 9 check acuraccy 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)
print("Test set results:")
print("Accuracy for reggression test:", metrics.accuracy_score(y_test, y_pred_test))
print("Accuracy for train reggression:", metrics.accuracy_score(y_train, y_pred_train))
#####################################################################################
logreg1 = LogisticRegression()
logreg1.fit(X_train1, y_train1)
y_pred_test1 = logreg1.predict(X_test1)
y_pred_train1 = logreg1.predict(X_train1)
print("Test set results:")
print("Accuracy for reggression test 2'set:", metrics.accuracy_score(y_test1, y_pred_test1))
print("Accuracy for train reggression 2'set:", metrics.accuracy_score(y_train1, y_pred_train1))
############################################################################################
cbc = CatBoostClassifier()
cbc.fit(X_train, y_train)
y_pred_test3 = cbc.predict(X_test)
y_pred_train3 = cbc.predict(X_train)
print("Test set results:")
print("Accuracy for catboost test:", metrics.accuracy_score(y_test, y_pred_test3))
print("Accuracy for train catboost:", metrics.accuracy_score(y_train, y_pred_train3))
##################################################################
cbc1 = CatBoostClassifier()
cbc1.fit(X_train1, y_train1)
y_pred_test4 = cbc1.predict(X_test1)
y_pred_train4 = cbc1.predict(X_train1)
print("Test set results:")
print("Accuracy for catboost test 2'set:", metrics.accuracy_score(y_test1, y_pred_test4))
print("Accuracy for train catboost 2'set:", metrics.accuracy_score(y_train1, y_pred_train4))

"""#9 b) there is no underfitting ocerfitting in the data and results 
the first model catboost with the data that was filled with the with the first way (regression with Accuracy for catboost test 2'set: 0.8327157226045526
Accuracy for train catboost 2'set: 0.8572563559322034) 
"""
