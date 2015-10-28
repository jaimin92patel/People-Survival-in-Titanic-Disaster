import pandas as pd
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

def change_char(df):
	df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
	df["Embarked"] = df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)	
	return df

train_df = change_char(pd.read_csv("train.csv"))

j=0;
for i in train_df["Age"]:
    np.nan=train_df["Age"][j]
    if np.isnan(np.nan):
        train_df["Age"][j]=np.median(train_df["Age"])
    j=j+1

estimate = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

#Columns to be considered
columns = ["Fare", "Pclass", "Sex"]
labels = train_df["Survived"].values
features = train_df[list(columns)].values

#print features	
cross_value = cross_val_score(estimate, features, labels, n_jobs=-1).mean()
print("{0} -> ET: {1})".format(columns, cross_value))

#Print row and column value
#Fix the value
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)

#imp
#Replace the Non Nmeric Value from Test.csv 
replaceDef = change_char(pd.read_csv("test.csv"))

#print test_df
estimate.fit(features, labels)

#Make Prediction for Survival
predictions = estimate.predict(imp.transform(replaceDef[columns].values))

#Store survival Result
replaceDef["Survived"] = pd.Series(predictions)
replaceDef.to_csv("final_result.csv", cols=['PassengerId', 'Survived'], index=False)