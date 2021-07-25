#Apriori*-
"""
Created on Sun Mar 28 14:34:25 2021

@author: Aiyub
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#data preprocesing
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
    
# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)
print()