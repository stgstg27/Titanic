# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 12:20:29 2017

@author: Saurabh
"""

from __future__ import print_function
import csv
import pandas as pd
from math import ceil
import numpy as np
from sklearn import tree


def max_three( a ,  b,  c):
    if(a>b and a>c):
        return a
    elif(b>a and b>c):
        return b
    elif(c>a and c>b):
        return c

#from tree import DecisionTreeClassfier

train =pd.read_csv("trux.csv")
#print (train.head())

#print (train.shape)
x = 0
list_of_missing_age = []



missing_age_filler = train["Age"].mean()
##missing_age_filler = ceil(missing_age_filler)

#print (missing_age_filler) :: Missing_age_filter =  29.61...

Train2 =np.matrix(train)

                                                              
#print (5 , end='\n\n')#Something Learnt during Experiment :P (:-) 
#print ('6',end='')

#print 5
#print 6
    


train["Age"] = train["Age"].fillna(missing_age_filler)
#train["Age"][train["Age"].isnull()==True] = missing_age_filler  ## This is method which work same as fillna method devied by me. 
                                                                 #It's advantage is it can work on Values which are None also Chill Enjoy
#print (train["Age"])
#train



#Filling out the missing value in Embarked
#First we find out the max of the 3 possible values and replace it with the same 
# replace the missing value with the value of max_value :P
#learned function::value_counts() 
embarked_S = train["Embarked"].value_counts()
embarked_value_S =  (embarked_S['S'])
embarked_value_Q =  (embarked_S['Q'])
embarked_value_C =  (embarked_S['C'])

max_value = max_three(embarked_value_S,embarked_value_Q,embarked_value_C)

if(max_value==embarked_value_S):
    train["Embarked"] = train["Embarked"].fillna("S")


if(max_value==embarked_value_C):
    train["Embarked"] = train["Embarked"].fillna("C")
    

if(max_value==embarked_value_Q):
    train["Embarked"] = train["Embarked"].fillna("Q")
    
    
#print (max_value)

#print (train["Embarked"])

# Pre-Processing the Value of Embarked Such That
#S:0 , C:1 ,  Q:2

train["Embarked_value"] = 0

train["Embarked_value"][train["Embarked"]=="C"] = 1
train["Embarked_value"][train["Embarked"]=="Q"] = 2

the_D_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

#print (train.head())

#Pre-Procseeing the Value of Sex
#Male:0
#Female:1

train["Sex_Number"] = 0
train["Sex_Number"][train["Sex"]=="female"] = 1


features = train[["Sex_Number" , "Age" , "Pclass","Embarked_value"]].values
target = train["Survived"].values


the_D_tree = the_D_tree.fit(features,target)

test = pd.read_csv("C:\Users\Saurabh\Downloads/test.csv")

#Pre Processing the Test data set  
missing_age_filler = test["Age"].mean()
test["Age"] = test["Age"].fillna(missing_age_filler)

test["Sex_Number"] = 0
test["Sex_Number"][test["Sex"]=="female"] = 1

test["Embarked_value"] = 0

test["Embarked_value"][test["Embarked"]=="C"] = 1
test["Embarked_value"][test["Embarked"]=="Q"] = 2

test_features = test[["Sex_Number" , "Age" , "Pclass","Embarked_value"]].values

test.Fare[152] = test["Fare"].median() 

target = the_D_tree.predict(test_features).astype(int)

PassengerId = np.array(test["PassengerId"]).astype(int)

my_sol = pd.DataFrame(target,PassengerId,columns = ["Survived"])

my_sol.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
#my_sol["PassengerId"] = test["PassengerId"]

#print (test["PassengerId"])
#ans = csv.DictWriter(my_sol ,fieldnames = ["PassengerId","Survivied"] )

#print (5)
#print (my_sol)

#my_sol_matrix = np.matrix(my_sol)

#print (my_sol_matrix)

#for i in my_sol:
#       print (i)
""""

with open('answer.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = ["Survival"])
    writer.writeheader()
    for i in range(len(my_sol)):
         writer.writerow(my_sol_matrix[i]) 
         
        

"""







#print (the_D_tree.score(features,target))

#print (the_D_tree.feature_importances_)


#print (543534534875348975893)


#for i in Train2:
    
        
    
    




