import glob
import pandas as pd
import numpy as np
import os
from engine.OutlierMeasures import *
from datetime import date,timedelta
import time

class modeling_preparation():
    
    """class to read the clead dataset and add outliers metrics
    
       INPUT: None
       OUTPUT: dataframe"""
       
    
    def __init__(self):
        self.today = date.today().strftime("%d/%m/%Y")

    def read_cleaned_df(self):    
        """function to idenitify and read the latest clean file

           INPUT: None
           OUTPUT: dataframe"""
        t0= time.time()

        # get list of cleaned files
        list_of_files = glob.glob('datasetFraud*.csv') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)

        # read most recent file
        print("file that will be used for this run is: {}".format(latest_file))
        print("reading file...")
        data = pd.read_csv(latest_file)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # fill in null values
        print("filling N/A...")
        data.fillna(0,inplace=True)
        display(data.head())

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data
    
    
    def run(self):
        data = self.read_cleaned_df()
        
        om = outlier_measures(data)
        data = om.run()
        
        return data