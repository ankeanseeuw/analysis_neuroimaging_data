# -*- coding: utf-8 -*-
"""
Anke Anseeuw

Psychological Perturbation Data on Attitudes Towards the Consumption of Meat

"""
# Import modules
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from  sklearn.model_selection import train_test_split

# Load csv file
filename='C:/Users/ankea/Documents/Psychologie/Master/2de master/CAED/script/PerturbationData.csv'
meat=pd.read_csv(filename,error_bad_lines=False, engine="python",sep=';')

# Preprocess
# Select only data we need
# df2 will contain data of only the first row of each participant (before perturbation)
meat1=meat.loc[[0],:]
x=1
while x<360:
    if x%12==0:
        df1=meat.loc[[x],:]
        meat1=pd.concat([meat1,df1])
    x=x+1


# Create 2 groups: carnivores and flexy- and vegetarians 
# We want to check whether a model can firstly correctly predict participants meat-eating based on the answers of the first questionnaire
# Thereafter, we want to check if it can still predict that well after all the perturbations, or did participants really change their minds

r=0
while r<30:
    if meat1.iloc[r,29]=='Omnivore':
        meat1.iloc[r,29]=0
    else:
        meat1.iloc[r,29]=1
    r=r+1   


 # of each lifestyle, there are 15 participants
       
# x and y 
 X=meat1.drop(['lifestyle', 'Perturbation','ï»¿Participant','d_scenario','gender'], axis=1)#bepalen welke ik nog wegdoe
 y=meat1['lifestyle']
 
 # Train and test splitting data
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random state opzoeken
 #nog eens opzoeken wat Frederik zei over verschillende splits within ofzo
 
 
 # apply standard scaler
 sc=StandardScaler()
 X_train=sc.fit_transform(X_train)
 X_test= sc.transform(X)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 