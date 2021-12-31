import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ function to read the csv files and assign them to variables
    
    INPUT: messages and categories paths and files
    OUTPUT: variables for messages and categories dataframes
    
    """
    # import csv into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df


def clean_data(df):
    """function to clean the merged dataframe. 
    the categories column has to be expanded into multiple columns with their respective rating for each message
    
    INPUT: merged dataframe
    OUTPUT: clean dataset
    """
    
    cat = df['categories'].str.split(";", expand=True)
    
    # below we extracted the names of the measures from the first row 
    # by taking the string apart from the last 2 characters from the columns
    row = cat.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    
    # assigning the new column names to the cat dataframe 
    cat.columns = category_colnames
    
    # take the value for each row and column and convert it to an integer
    for column in cat:
        # set each value to be the last character of the string
        cat[column] = cat[column].astype(str).str[-1]

        # conversion - convert column from string to numeric
        cat[column] = cat[column].astype(int)
    #cat.head()
    
    # merge the transformed cat dataframe to the actual dataframe, drop useless column and return the final clean dataframe
    df = pd.concat([df,cat], axis=1)
    df.drop(columns='categories',inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    # Remove rows with a  value of 2 from df
    df = df[df['related'] != 2]
    
    return df


def save_data(df, database_filename):
    
    """Save into  SQLite database.
    
            INPUT: df: dataframe. Dataframe containing cleaned version of merged message and categories data.
                   database_filename: string. Filename for output database.

            OUTPUT:None
    """
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.split('/')[-1]
    table_name = table_name.split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    
    print("saved dataframe table in {} database as 'Messages'".format(database_filename))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()