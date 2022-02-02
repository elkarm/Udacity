# Capstone-Project-Fraud-Detention-Udacity
-----------------------------------------------------------------------------------------------------------------------------------

## Project Overview
For the Project Capstone for the Data Science Nanodegree in Udacity we examine how we can identify fraudulent transactions based on a simulated dataset of fraudulent and geniune transactions provided in KAGGLE.

## Problem Statement
The question on how to identify fraudulent transactions gains more and more ground over time. From the financial sector to marketing and product companies, there is a need to understand how to identify transactions that are not genuine.
There maybe basic rules depenting on the reason and the sector, however here we will investigate how to identify a fraudulent transaction with the use of a basic dataset that was created by PaySim application. To identify the fraudulent 
transactions we use 4 different models to see if any of them can identfy the frauds among all the transactions, compare these models and keep the best performing one.

## Note on metrics
Typically, when we look at the results of a classification model, we focus on the correct predictions of all the predictions made by the model, i.e., the accuracy of the model. 
In our case, when doing a churn analysis with the goal of predicting customers who churned, we are particularly interested in having a lot of true positives. However, since there are many customers who have not churned (highly unbalanced datasets), the higher the number of true negatives the higher the accuracy, which can be misleading. Thus, a better measure of model performance in this case is the F1 score, which is the harmonic mean of Precision ('out of all customers who were labeled as "churned," how many did we correctly label as such?') and Recall ('out of all customers who were labeled as "churned," how many actually churned?').

## Data
Unfortunately due to the large volume of data the file cannot be saved in GitHUB. for this reason, please navigate to kaggle (https://www.kaggle.com/ealaxi/paysim1).

The dataset contains a snthetic dataset with basic transactional information and a flag to identify which transactions are fraudulent.
example data

step     type      amount     nameOrig   oldbalanceOrg   newbalanceOrig    nameDest    oldbalanceDest   newbalanceDest   isFraud  isFlaggedFraud

  1     PAYMENT   1060.31    C429214117     1089.0            28.69       M1591654462        0.0               0.0           0           0

 --> step = maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
 
 --> type = CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
 
 --> amount = amount of the transaction in local currency.
 
 --> nameOrig = customer who started the transaction
 
 --> oldbalanceOrg = initial balance before the transaction
 
 --> newbalanceOrig = new balance after the transaction
 
 --> nameDest = customer who is the recipient of the transaction
 
 --> oldbalanceDest = initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
 
 --> newbalanceDest = new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
 
 --> isFraud = This is the transactions made by the fraudulent agents inside the simulation. 
 
               In this specific dataset the fraudulent behavior of the agents aims to profit by taking control 
               
                   or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
                   
 --> isFlaggedFraud = The business model aims to control massive transfers from one account to another and flags illegal attempts. 
 
                      An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.


## Installation
Jupyter notebook
Python3
Numpy, Pandas
Data Visualization: Plotly, Seaborn, Matplotlib
sklearn, pickle, scipy

Instructions for running the notebook:



Results
The best performing model is chosen based on a compination of the accuracy, precision, recal, F1 and cross-validation metrics. The best performing model for the specific classification is the random forest classifier which showed very good results in all the metrics.
The jupiter notebook alo provides some exploratory analysis on the dataset with some insights to the data. Finally, we use this machine learning model trained on the full dataset and save it in a pickle file for future use.

## Licensing, Authors, Acknowledgements.
This project has been completed as part of the Data Science Nanodegree on Udacity. The data was provided by kaggle - https://www.kaggle.com/ealaxi/paysim1 - which i own a massive thanks.
