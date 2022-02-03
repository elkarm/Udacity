import pandas as pd
import numpy as np
from datetime import date,timedelta
import time
from scipy import stats


class outlier_measures():
    
    """class to calcualte and add to the dataset outlier measures as z_score and IQR to identify how many times
       each record is marked as an outlier based on these 2 measures
       
       INPUT: data = dataframe
              cols = columns to perform the outlier measures
              
       OUTPUT: dataframe with 2 new columns"""
    
    
    def __init__(self, data, cols=['amount','Error_oldbalanceOrig','Error_Orig_fnlBlnce','Error_oldbalanceDest','Error_Dest_fnlBlnce','origEntity_trxCnt','destEntity_trxCnt']):
        self.data = data
        self.cols = cols
        self.z_threshold = 3
        self.today = date.today().strftime("%d/%m/%Y")
        

    def z_score(self):
        
        """funtion to calculate cumulative sum of the columns marked as outliers due to z-score for the selected columns
        
           INPUT: None
           OUTPUT: dataframe with sum of outlier measures"""
        
        t0= time.time()


        # calculate z score
        print("calculate z-score...")
        z = np.abs(stats.zscore(self.data[self.cols]))

        # rename cols in the z dataframe
        cols_z=['{}_z'.format(c) for c in self.cols]
        zdf=pd.DataFrame(z,columns=cols_z)

        # if at least one of the selected characteristics has z-score>3 then the whole row is marked as an outlier
        print("creating flag to identify outliers in the dataset...")
        zdf=zdf > self.z_threshold
        zdf['z_outlier'] = zdf.sum(axis=1)

        # merge outlier result to the x_train dataset
        self.data = self.data.merge( zdf[['z_outlier']], left_index=True, right_index=True)

        print("check z-score outlier result")
#         display(self.data.head())
#         display(self.data.groupby('z_outlier')['amount'].count())
        display(self.data.groupby(['z_outlier','isFraud'])['amount'].count())

        # clean up
        del z, cols_z,zdf        

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        
        return self.data
    
    
    def iqr_score(self):
        
        """funtion to calculate cumulative sum of the columns marked as outliers due to IQR-score for the selected columns
        
           INPUT: None
           OUTPUT: dataframe with sum of outlier measures"""
        
        t0= time.time()

        # calculate Q1,Q3,IQR
        print("calculate Q1,Q3,IQR...")
        Q1 = self.data[self.cols].quantile(0.25)
        Q3 = self.data[self.cols].quantile(0.75)
        IQR = Q3 - Q1

        # IQR_outlier flag
        print("calculate iqr_outlier...")
        iqr_outliers = (self.data[self.cols] < (Q1 - 1.5 * IQR)) |(self.data[self.cols] > (Q3 + 1.5 * IQR))

        iqr_outliers['iqr_outlier'] = iqr_outliers.sum(axis=1)

        # merge outlier result to the x_train dataset
        print("merge outlier result to data...")
        self.data = self.data.merge( iqr_outliers[['iqr_outlier']], left_index=True, right_index=True)

        print("check iqr outlier result")
#         display(self.data.head())
#         display(self.data.groupby('iqr_outlier')['amount'].count())
        display(self.data.groupby(['iqr_outlier','isFraud'])['amount'].count())

        # clean up
        del Q1, Q3,IQR,iqr_outliers

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        
        return self.data
    
    
    def run(self):
        self.data = self.z_score()
        self.data = self.iqr_score()
        return self.data