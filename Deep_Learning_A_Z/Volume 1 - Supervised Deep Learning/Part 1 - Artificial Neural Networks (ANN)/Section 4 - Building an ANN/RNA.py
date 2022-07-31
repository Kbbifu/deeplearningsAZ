# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Réseau des neurones artificiels
#partie 1 : Préparation des données

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.drop('customerID',axis='columns',inplace=True)

print(dataset.dtypes)
print(dataset.TotalCharges.values)

#pd.to_numeric(dataset.TotalCharges)

print(pd.to_numeric(dataset.TotalCharges,errors='coerce').isnull())
print(dataset[pd.to_numeric(dataset.TotalCharges,errors='coerce').isnull()])

print(dataset.shape)
dataset.iloc[488].TotalCharges
print(dataset[dataset.TotalCharges!=' '].shape)

#Remove rows with space in TotalCharges

df1 = dataset[dataset.TotalCharges!=' ']
print(df1.shape)

print(df1.dtypes)

df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
print(df1.TotalCharges.values)
print(df1['Churn'].value_counts())

#Data Visualization

tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")


plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")


plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()

#Many of the columns are yes, no etc. Let's print unique values in object columns to see data values

def print_unique_col_values(dataset):
       for column in dataset:
            if dataset[column].dtypes=='object':
                print(f'{column}: {dataset[column].unique()}')

print(print_unique_col_values(df1))


#Some of the columns have no internet service or no phone service, that can be replaced with a simple No

df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)

print(print_unique_col_values(df1))

#Convert Yes and No to 1 or 0

yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)

for col in df1:
    print(f'{col}: {df1[col].unique()}') 

df1['gender'].replace({'Female':1,'Male':0},inplace=True)
print(df1.gender.unique())

#One hot encoding for categorical columns

df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
print(df2.columns)

print(df2.sample(5))
print(df2.dtypes)

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

for col in df2:
    print(f'{col}: {df2[col].unique()}')
    

#Train test split

X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

#Build a model (ANN) in tensorflow/keras

import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)

yp = model.predict(X_test)
print(yp[:5])

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
        
print(y_pred[:10])

from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))

import seaborn as sns
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(y_test.shape)

#Accuracy

print(round((862+229)/(862+229+137+179),2))

#Precision for 0 class. i.e. Precision for customers who did not churn

print(round(862/(862+179),2))

#Precision for 1 class. i.e. Precision for customers who actually churned

print(round(229/(229+137),2))

#Recall for 0 class

print(round(862/(862+137),2))

print(round(229/(229+179),2))



