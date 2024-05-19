import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("..../train.csv")

train.info()

# count number of categorical and numerical columns
train = train.drop(columns=['Loan_ID'])
categorical_columns = [var for var in train.columns if train[var].dtype == 'object']
numerical_columns = [var for var in train.columns if train[var].dtype != 'object']

# categorical columns
fig, axes = plt.subplots(4,2,figsize=(12,15))
for idx, cat_col in enumerate(categorical_columns):
    row, col = idx//2, idx%2
    sns.countplot(x=cat_col, data=train, hue='Loan_Status',ax=axes[row,col])
    
plt.subplots_adjust(hspace=1)

# numerical columns
fig, axes = plt.subplots(1,3, figsize=(17,5))
for idx, cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col, data=train, x='Loan_Status', ax=axes[idx])
    
print(train[numerical_columns].describe())
plt.subplots_adjust(hspace=1)

# encoding categorical features :
train_encoded = pd.get_dummies(train, drop_first=True)
train_encoded.head()

# split features and target variable

x = train_encoded.drop(columns='Loan_Status_Y')
y = train_encoded['Loan_Status_Y']

# splitting into Train -Test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify = y,random_state=42)

# handling / imputing missing values 
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(x_train)
x_train = imp_train.transform(x_train)
x_test_imp = imp_train.transform(x_test)

# model 1 : decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score 

tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_train, y_train)
y_pred = tree_clf.predict(x_train)

print("Training Data Set Accuracy: ", accuracy_score(y_train,y_pred))
print("Training Data F1 Score ", f1_score(y_train,y_pred))

print("Validation Mean F1 Score: ",cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='accuracy').mean())

# tunning hyperparameter

training_accuracy = []
val_accuracy = []
training_f1 = []
val_f1 = []
tree_depths = []

for depth in range(1,20):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(x_train,y_train)
    y_training_pred = tree_clf.predict(x_train)

    training_acc = accuracy_score(y_train,y_training_pred)
    train_f1 = f1_score(y_train,y_training_pred)
    val_mean_f1 = cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='accuracy').mean()
    
    training_accuracy.append(training_acc)
    val_accuracy.append(val_mean_accuracy)
    training_f1.append(train_f1)
    val_f1.append(val_mean_f1)
    tree_depths.append(depth)
    
Tuning_Max_depth = {"Training Accuracy": training_accuracy, "Validation Accuracy": val_accuracy, "Training F1": training_f1, "Validation F1":val_f1, "Max_Depth": tree_depths }
Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)

plot_df = Tuning_Max_depth_df.melt('Max_Depth',var_name='Metrics',value_name="Values")
fig,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Max_Depth", y="Values",hue="Metrics", data=plot_df,ax=ax)

# Visualizing Decision Tree with max depth = 3

import graphviz 
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(max_depth = 3)
tree_clf.fit(x_train,y_train)
dot_data = tree.export_graphviz(tree_clf,feature_names = x.columns.tolist())
graph = graphviz.Source(dot_data)
graph