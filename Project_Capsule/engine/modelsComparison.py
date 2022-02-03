import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
import pickle
from datetime import date,timedelta
import time
from scipy import stats
from sklearn import metrics
# show all columns
pd.set_option("display.max_columns", None)
# deactivate scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class models_comparison():
    
    """class to split and compare how different model perform on the dataset
       
       IPNUT: data = dataframe for analysis
              models_to_compare = dictionary of models and their names to be used for comparison so the best model to be choosen by the user
              
       OUTPUT: a results table to show the models fit on the model and their performance"""
    
    def __init__(self,data, models_to_compare,target,exclude_x_cols = ['type','isFlaggedFraud', 'isFraud']):
        self.data = data
        self.models_to_compare = models_to_compare
        self.model_rslts = pd.DataFrame(columns = ['model','accuracy','precision','recall','f1', 'cv_precision'])
        self.today = date.today().strftime("%d/%m/%Y")
        self.exclude_x_cols = exclude_x_cols
        self.target = target
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_into_train_test(self.data)
        
        
    def split_into_train_test(self,data):
        """method to split data to training/test and indepented and target variables
        
           INPUT: data = dataframe to split
                  target = the target variable
                  exclude_x_cols = any extra columns to exclude from the variables
                  
            OUTPUT: split data : x_train, x_test, y_train, y_test"""
        t0= time.time()

        # split the dataset to the variables and the target
        print("spliting data to variables(x) and target(y)...")
        # include columns that are not in the excluded and not object data types
        x = data[[col for col in data.columns if col not in self.exclude_x_cols + list(data.dtypes[data.dtypes == np.object].index)]]
        y = data[self.target]

        # split x,y to train and test now so there is no data leceage later in the script
        print("spliting data to train and test...")
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return x_train, x_test, y_train, y_test



    
    def model_eval(self,title, model,x_train,y_train,x_test, y_test):
        #,model_rslts = None
        
        """Method to evaluate the model. It fits the model, and create metrics to evaluate the performance
        
           INPUT: title = title of the model
                  model = the model class to be used
                  model_rslts = dataframe to save the model accuracy metrics results for comparison
                  
           OUTPUT: prints steps in the process and the accuracy results""" 
            
        t0 = time.time()
        
#         if model_rslts is None:
#             model_rslts = self.model_rslts
        
        
        # Instantiate
        print("\n\ninstansiate model '{}' ...".format(title))
        model_inst = model   

        # Fit
        print("model fitting...")
        model_fit = model_inst.fit(x_train, y_train)

        # accuracy
        print("model accuracy for trained data = {}".format(model_fit.score(x_train, y_train)))


        # show the importance of the variables used for the model
        try:
            # get importance
            print("get importance...")
            importance = model.feature_importances_
            
            imps = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), x_train), reverse=True)
            imps_df = pd.DataFrame(imps,columns=['importance','feature'])
            imps_df.sort_values(by='importance', ascending=False)

            fig = plt.figure(figsize = (14, 9))
            imps_df.plot.barh(x='feature',y='importance')
            plt.title("variables per importance for the model")
            plt.show()
            
        except:
            print("importance function diffes for the specific model")

        # Predictions/probs on the test dataset
        print("predictions and probs on test dataset...")
        predicted = pd.DataFrame(model_fit.predict(x_test))
        probs = pd.DataFrame(model_fit.predict_proba(x_test))

        # Store metrics
        print("calculate metrics...")
        accuracy = metrics.accuracy_score(y_test, predicted)
        roc_auc = metrics.roc_auc_score(y_test, probs[1])
        confus_matrix = metrics.confusion_matrix(y_test, predicted)
        classification_report = metrics.classification_report(y_test, predicted)
        precision = metrics.precision_score(y_test, predicted, pos_label=1)
        recall = metrics.recall_score(y_test, predicted, pos_label=1)
        f1 = metrics.f1_score(y_test, predicted, pos_label=1)

        # Evaluate the model using 10-fold cross-validation
        print("calculate cross-validation...")
        cv_scores = cross_val_score(model, x_test, y_test, scoring='precision', cv=10)
        cv_mean = np.mean(cv_scores)

        print("Store metrics...")
        self.model_rslts = self.model_rslts.append({'model':title
                               ,'accuracy':accuracy
                               ,'precision':precision
                               ,'recall':recall
                               ,'f1':f1
                               ,'cv_precision':cv_mean },ignore_index=True)

        self.model_rslts.sort_values(by='precision', ascending=False)

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        
    def run(self):
        
        for title, model in self.models_to_compare.items():
            self.model_eval(title, model,self.x_train,self.y_train,self.x_test, self.y_test)
            
        display(self.model_rslts.sort_values(by='precision', ascending=False))
        
    
    def run_save_model(self,model_params):
        
        t0 = time.time()
        
        print("CHOSEN MODEL: {title}".format(**model_params))
        
        print("# Create x and y from all data...")
        x = self.data[[col for col in self.data.columns if col not in self.exclude_x_cols + list(self.data.dtypes[self.data.dtypes == np.object].index)]]
        y = self.data[self.target]
        
        print("# Re-train model on all data...")
        model = model_params['model'].fit(x, y)
        
        print("# Save model pickle file classifier_{}.pkl".format(date.today().strftime("%Y%m%d")))
        with open('classifier_{}.pkl'.format(date.today().strftime("%Y%m%d")), 'wb') as fid:
            pickle.dump(model, fid)
            
        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        