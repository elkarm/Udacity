from IPython.core.display import display, HTML
from itertools import chain,cycle
from IPython.display import display_html
import pandas as pd
import numpy as np

# useful functions

# stats
def describe(df, stats = ['skew', 'mad', 'kurt', 'sum']):
    """function to add more statitstical fields in the discribe function
    
       INPUT: df = dataframe to analyse
              stats = list of statistical measures to add
              
       OUTPUT: dataframe with stats for the numric columns of the dataset"""
    
    d = df.describe()
    return d.append(df.reindex(d.columns, axis = 1).agg(stats))


# utils
def display_side_by_side(*args,titles=cycle([''])):
    """function to display 2 or more tables in parallel
    
       INPUT: *args = dataframes
              titles: list of titles for each table prineted in the row
       
       OUTPUT: parallel display of dataframes"""

    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)