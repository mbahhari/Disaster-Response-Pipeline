# Disaster-Response-Pipeline

In this project a machinle learning application will be applied by importing a dataset that contains a real message from real events.
This application will analyse the messages inserted to decide which agency should receive the message.

# Three steps will be taken in this procject 
## 1- Data cleaning 
- load data :load two csv files messages and categories and merging two files base on id
 - clean data : create a dataframe of the 36 individual category columns , drop duplicates
 - save data : save data to a database 
 run the code : to run the code all arguments should be defined 
 For example : python process_data.py messages.csv categories.csv DisasterResponse.db 
 where DisasterResponse.db is the name of the saved database

## 2- Train the data :
- load the data from the database file
- Creat a tokenize function and apply it on the text
- split the data to training data set and test dataset
- build a model using pipeline and applying RandomForestClassifier 
- train and evaluate the model the model
- save the model to classifier.pk1 
- run the code  python train_classifier.py ../data/DisasterResponse.db classifier.pk1 
- the classifier.pk1  will be creatd

## 3- run the application
the web applicaction is already set
- we need to defined the path of the database and the model
- run the application : python run.py 


