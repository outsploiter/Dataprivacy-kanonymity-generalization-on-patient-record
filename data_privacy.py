#basic lib to work with dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#libraries to work with the anonymity of the proc(Data)
from cn.protect import Protect
from cn.protect.privacy import KAnonymity
from cn.protect.hierarchy import DataHierarchy, OrderHierarchy, IntervalHierarchy
from cn.protect.quality import Loss #to calculate the loss of the data


dataset=pd.read_csv("dataset/raw_data1.csv")
print(dataset.dtypes)
print(dataset.head())
print(dataset.isnull().any())

#filling the NaN or null values with median or mode of the values
dataset['Patient Number'].fillna(dataset['Patient Number'].median(),inplace = True)
dataset['State Patient Number'].fillna(dataset['State Patient Number'].mode()[0],inplace = True)
dataset['Age Bracket'].fillna(dataset['Age Bracket'].mode()[0],inplace = True)
dataset['Gender'].fillna(dataset['Gender'].mode()[0] ,inplace = True)
dataset['Detected City'].fillna(dataset['Detected City'].mode()[0],inplace = True)
dataset['Detected District'].fillna(dataset['Detected District'].mode()[0],inplace = True)
dataset['Detected State'].fillna(dataset['Detected State'].mode()[0],inplace = True)
dataset['State code'].fillna(dataset['State code'].mode()[0],inplace = True)
dataset['Nationality'].fillna(dataset['Nationality'].mode()[0] ,inplace = True)
dataset['Type of transmission'].fillna(dataset['Type of transmission'].mode()[0] ,inplace = True)
dataset['Status Change Date'].fillna(dataset['Status Change Date'].mode()[0] ,inplace = True)
dataset.drop(['Source_1', 'Source_2', 'Source_3', "Contracted from which Patient (Suspected)", "Estimated Onset Date"], axis=1, inplace=True)
dataset["Age Bracket"].replace({"28-35": "32", "1.5": 2}, inplace=True)
dataset["Age Bracket"] = dataset["Age Bracket"].astype(str).astype(int)
print(dataset.tail(2))

#labelEncoding the Patient's identity
dataset["Patient Number"]=le.fit_transform(dataset["Patient Number"])
dataset["State Patient Number"]=le.fit_transform(dataset["State Patient Number"])
print(dataset.head())


#visualizing the dataset
import seaborn as sns
print(sns.pairplot(dataset))

#applying KAnonymity, suppression, loss functions on the data by creating a prot datatype
prot=Protect(dataset, KAnonymity(17300))
prot.quality_model=Loss()
prot.suppression=.1

#hiding the identifiers (explicit)
for col in dataset:
    if col not in ("Patient Number", "State Patient Number", "Detected District"):
        prot.itypes[col]='insensitive'

prot.itypes["Patient Number"]='identifying'
prot.itypes["State Patient Number"]='quasi'
prot.itypes["Detected District"]='quasi'
prot.itypes["Age Bracket"]='insensitive'
print(prot.itypes)


#transfering prot data type to dataframe(priv)
priv = prot.protect()

#generalizing the age
priv = prot.protect()
priv=priv.rename(columns={"Age Bracket":"age"})
bins = [0,18, 30, 40, 50, 60, 70, 120]
labels = ['0-17','18-29', '30-39', '40-49', '50-59', '60-69', '70+']
priv['Age'] = pd.cut(priv.age, bins, labels = labels,include_lowest = True)
priv["age"]=priv["Age"]
priv.drop(["Age"], axis=1, inplace=True)


#saving dataframe to csv file
dataset.to_csv('Privacy_Protected_rawdata1.csv',index=False)
