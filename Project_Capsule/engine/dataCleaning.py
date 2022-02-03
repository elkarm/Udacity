# labrary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date,timedelta
import time

today = date.today().strftime("%d/%m/%Y")

# deactivate scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from engine.utilities import describe,display_side_by_side

# expand cells
from IPython.core.display import display

class data_cleaning():
    
    """class with all the functions to clean the dataset"""
    
    def __init__(self):
        self.today = date.today().strftime("%d/%m/%Y")
        
    
    def data_reading(self,file):
        """
        function to read and discride dataset

        INPUT: path to csv file

        OUTPUT: dataframe of file and infomation of the file
        """

        t0= time.time()
        print('data reading...')
        data = pd.read_csv('PS_20174392719_1491204439457_log.csv')
        print("dataset initial shape: {}".format(data.shape))
        display(data.head())
        print("show basic stats...")
        display(describe(data, ['skew', 'mad', 'kurt', 'sum']))
        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        return data


    def drop_miss_dublicates(self,data):
        """function to clean data from missing and duplicate values as they can create noise in the results

           INPUT: dataframe to check

           OUTPUT: cleaned dataset after removing missing values and dropping duplicates
           """

        t0= time.time()

        # 1 record does not have target variable available. Drop this record as it adds up to nothing
        data = data.loc[(data['isFlaggedFraud'].notna())|(data['isFraud'].notna())]

        print("\nduplicate values can cause noise in the data. Thus checking if there are any and dropping them")
        z = data.duplicated().value_counts()
        if True in z:
            print("there are duplicates that are dropped")
            df.drop_duplicates(inplace = True)
        else:
            print("no duplicates")

        del z

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data


    def cust_merch_ind(self,data):
        """function to extract Customer or Merchant indicator from Originator and Destination

           IPNUT: dataframe

           OUTPUT: dataframe with extra columns"""

        t0= time.time()

        print("\ngetting the Customer and Merchant indicators")
        data['ext_org_ind'] = data['nameOrig'].str[:1]
       # display(data[['ext_org_ind','nameOrig']].groupby('ext_org_ind').agg({'nameOrig':['count','nunique']}))
        data['ext_den_ind'] = data['nameDest'].str[:1]
       # display(data[['ext_den_ind','nameDest']].groupby('ext_den_ind').agg({'nameDest':['count','nunique']}))

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data






    def check_balances(self,direction, old_balance, new_balance, data):

        """function to check and """

        t0= time.time()
        print("\nCheck balances for {}...".format(direction))

        # when old and new balalnce are marked as 0 this indicated an error or lack of information and thus we mark it as -1 in the data to recognise them easier
        print("update old and new balances to reflect missing data...")
        data['Error_oldbalance{}'.format(direction)]  = np.where((data[old_balance] == 0)&(data[new_balance] == 0)&(data['amount']!=0)  , -1, data[old_balance])
        data['Error_newbalance{}'.format(direction)] = np.where((data[old_balance] == 0)&(data[new_balance] == 0)&(data['amount']!=0)  , -1, data[new_balance])

        # -1 to avoid zeors as log doesnt work on 0 and -2 to recognise the -1 (missing values)
        data['log_Error_newbalance{}'.format(direction)] = np.where((data['Error_newbalance{}'.format(direction)] == -1)  , -2, 
                                                                    np.where(data['Error_newbalance{}'.format(direction)] == 0  , -1, np.log10(data['Error_newbalance{}'.format(direction)])))

        # when the tran type is Cash In then it is credit transaction for the roiginator that increases their final amount
        data['Error_{}_fnlBlnce'.format(direction)] = np.where((data[old_balance] == 0) & (data[new_balance] == 0) & (data['amount']!=0)  , -1 ,
                                               np.where(data['type']=='CASH_IN', data[old_balance] + data['amount'], data[old_balance] - data['amount']))

        # calculate the difference of the existing and re-calculated balances
        print("calculate difference of existinct and re-calcualted final balances...")
        data['diff_{}_fnl_balance'.format(direction)] = data['Error_{}_fnlBlnce'.format(direction)] - data[new_balance]

        conditions = [ # are equal or recalc_newbalance_org-1<=newbalanceOrig<=recalc_newbalance_org+1 - weird roundings
        (  data['Error_{}_fnlBlnce'.format(direction)] == np.ceil(data[new_balance])),
        ( (data['Error_{}_fnlBlnce'.format(direction)] == -1) & (np.ceil(data[new_balance]) == np.ceil(data[old_balance]) ) ),
        ( (data['Error_{}_fnlBlnce'.format(direction)]!= -1) & (np.ceil(data[new_balance]) != np.ceil(data[old_balance]) ) )
                 ]
        
        choices = ['correct - equals', 'missing data - not talking negative values', 'wrong']
        data['recalc_vs_df_newbalance_{}'.format(direction)] = np.select(conditions, choices, default='missing')
        display(data[['recalc_vs_df_newbalance_{}'.format(direction),'isFraud']].groupby(['recalc_vs_df_newbalance_{}'.format(direction)]).agg({'isFraud':['count','sum']}))


        display(data[['recalc_vs_df_newbalance_{}'.format(direction),'type','isFraud','amount',new_balance]].loc[data['type'].isin(['CASH_OUT','TRANSFER'])].groupby(['recalc_vs_df_newbalance_{}'.format(direction), 'type']).agg({'isFraud':['count','sum']
                                                                                                                                                                         ,'amount':['min','max']
                                                                                                                                                                         ,new_balance:['min','max']}))



        # filter the data fro distributions and other graphs
        sns.catplot(x="isFraud", y="log_Error_newbalance{}".format(direction), data=data.loc[data['type'].isin(['CASH_OUT','TRANSFER'])], aspect=2)
        sns.catplot(x="isFraud", y="log_Error_newbalance{}".format(direction), data=data.loc[data['type'].isin(['CASH_OUT','TRANSFER'])], aspect=2)

        print("distributions...")
        Fraud_cshout_transfer = data.loc[(data['type'].isin(['CASH_OUT','TRANSFER']))]

        print("NO FRAUD")
        df = Fraud_cshout_transfer.loc[Fraud_cshout_transfer['isFraud']==0]['diff_{}_fnl_balance'.format(direction)]
        df.hist(bins=40, figsize=(10, 5), sharex=False, sharey=False, by=Fraud_cshout_transfer['type'])
        plt.ticklabel_format(style='plain')
        plt.legend(prop={'size': 10})
        plt.show()

        print("FRAUD")
        Fraud_cshout_transfer.loc[Fraud_cshout_transfer['isFraud']==1]['diff_{}_fnl_balance'.format(direction)].hist(bins=40, figsize=(10,5), sharex=False, sharey=False, by=Fraud_cshout_transfer['type'])
        plt.ticklabel_format(style='plain')
        plt.legend(prop={'size': 10})
        plt.show()

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))
        return data



    # orig entity is originator x amoount of transactions

    def orig_entity_position_freq(self,data):

        """function to get the frequesny each originator entity appearns as originator and destination

           IPNUT: dataframe

           OUTPUT: dataframe with extra columns"""

        t0= time.time()

        print("ORIG ENTITY: originator frequencies...")
        nameOrig_freq = data.groupby(['nameOrig'])['nameOrig'].count().reset_index(name="origEntity_trxCnt_origPos")
        # fill na with 0 as it indicates no transactions
        nameOrig_freq.fillna(0,inplace=True)
        # merge to dataset
        data = data.merge(nameOrig_freq,how = 'left', on = 'nameOrig')
        # rename to prepare for the total count per customer
        nameOrig_freq.rename(columns={'nameOrig':'name', 'origEntity_trxCnt_origPos':'origEntity_trxCnt'},inplace = True)

        print("ORIG ENTITY: destination frequencies...")
        nameDest_freq = data.groupby(['nameDest'])['nameDest'].count().reset_index(name="origEntity_trxCnt_destPos")
        # fill na with 0 as it indicates no transactions
        nameDest_freq.fillna(0,inplace=True)
        # merge to dataset
        data = data.merge(nameDest_freq,how = 'left',left_on = 'nameOrig', right_on = 'nameDest').drop(columns= ['nameDest_y']).rename(columns={'nameDest_x':'nameDest'})
        # rename to prepare for the total count per customer
        nameDest_freq.rename(columns={'nameDest':'name', 'origEntity_trxCnt_destPos':'origEntity_trxCnt'},inplace = True)

        print("add originator frequencies to final table...")
        all_Origcust_freq = nameOrig_freq.append(nameDest_freq)
        all_Origcust_freq = all_Origcust_freq.groupby('name')['origEntity_trxCnt'].sum().reset_index(name="origEntity_trxCnt")
        # merge to dataset
        data = data.merge(all_Origcust_freq,how = 'left', left_on = 'nameOrig', right_on = 'name').drop(columns= ['name'])

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data




    # Destination entity is destination for x amoount of transactions

    def dest_entity_position_freq(self,data):

        """function to get the frequesny each Destination entity appearns as originator and destination

           IPNUT: dataframe

           OUTPUT: dataframe with extra columns"""

        t0= time.time()  

        print("DEST ENTITY: originator frequencies...")
        DnameDest_freq = data.groupby(['nameDest'])['nameDest'].count().reset_index(name="destEntity_trxCnt_destPos")
        # fill na with 0 as it indicates no transactions
        DnameDest_freq.fillna(0,inplace=True)
        # merge to dataset
        data = data.merge(DnameDest_freq,how = 'left', on = 'nameDest')
        # rename to prepare for the total count per customer
        DnameDest_freq.rename(columns={'nameDest':'name', 'destEntity_trxCnt_destPos':'destEntity_trxCnt'},inplace = True)

        print("DEST ENTITY: destination frequencies...")
        DnameOrig_freq = data.groupby(['nameOrig'])['nameOrig'].count().reset_index(name="destEntity_trxCnt_origPos")
        # fill na with 0 as it indicates no transactions
        DnameOrig_freq.fillna(0,inplace=True)
        # merge to dataset
        data = data.merge(DnameOrig_freq,how = 'left',left_on = 'nameDest', right_on = 'nameOrig').drop(columns= ['nameOrig_y']).rename(columns={'nameOrig_x':'nameOrig'})
        # rename to prepare for the total count per customer
        DnameOrig_freq.rename(columns={'nameOrig':'name', 'destEntity_trxCnt_origPos':'destEntity_trxCnt'},inplace = True)

        print("add destination frequencies to final table...")
        all_Destcust_freq = DnameOrig_freq.append(DnameDest_freq)
        all_Destcust_freq = all_Destcust_freq.groupby('name')['destEntity_trxCnt'].sum().reset_index(name="destEntity_trxCnt")
        # merge to dataset
        data = data.merge(all_Destcust_freq,how = 'left', left_on = 'nameDest', right_on = 'name').drop(columns= ['name'])

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data



    def dummy_variables(self,data):

        t0= time.time()
        
        print("""add dummy variables of categorical columns...""")

        # dummy variables for type trx, exclude debit. when all are 0 then the trx is debit
        for col in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN']:
            data[col] = np.where(data['type']==col,1,0)

        # dummy variable for ext_org_ind & ext_den_ind
        data['ext_org_ind_C'] = np.where(data['ext_org_ind']=='C',1,0)
        data['ext_den_ind_C'] = np.where(data['ext_den_ind']=='C',1,0)

        # delete string variables
        #del data['type']
        del data['nameOrig']
        del data['nameDest']
        del data['ext_org_ind']
        del data['ext_den_ind']

        display(data.head())

        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))

        return data
    
    
    def save_clean_df(self,data):
        
        t0= time.time()
        
        print("save dataset...")
        
        data.to_csv("datasetFraud_variables_{}.csv".format(date.today().strftime("%Y%m%d")))
        print("file is saved as:  datasetFraud_variables_{}.csv".format(date.today().strftime("%Y%m%d")))
        
        print("date: {}   runtime: {}".format(self.today,str(timedelta(seconds=(time.time()-t0)))))