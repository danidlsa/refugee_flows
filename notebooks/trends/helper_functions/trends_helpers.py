import pandas as pd

from helper_functions.country_abbrev import *
from helper_functions.country_language import *
from pytrends.request import TrendReq

import os
import pycountry
import itertools

from googletrans import LANGCODES
import swifter
import requests
import numpy as np

import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets.embed import embed_minimal_html
import seaborn as sns
import matplotlib.pyplot as plt

def translate_keywords_series(series, lang):

    series = series.str.split('+').explode()
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "auto",
        "tl": lang,
        "dt": "t",
        "q": "\n".join(series.tolist())
    }
    response = requests.get(url, params=params)
    series_translated = [r[0].strip('\n').lower() for r in response.json()[0]]
    series_translated = pd.Series(index=series.index.tolist(), data=series_translated, name = series.name).to_frame().groupby(series.index)[series.name].agg(list).apply(lambda x: '+'.join(x))
    return series_translated


def translate_keywords_list(lst, lang):
    lst = [item.split('+') for item in lst]
    lst = [item for sublist in lst for item in sublist]
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "auto",
        "tl": lang,
        "dt": "t",
        "q": "\n".join(lst)
    }
    response = requests.get(url, params=params)
    lst_translated = [r[0].strip('\n').lower() for r in response.json()[0]]
    return lst_translated



# Function to get ISO2 country code
def get_iso2_country_code(country_name):
    print("Converting " + country)
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_2
    except LookupError:
        return None


def file_counter(folder_path:str):
    file_count = 0
    # Iterate over each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            file_count += 1

    return file_count


def get_min_max(row, data_frame, group_column1, group_column2, value_column):
    subgroup1 = row[group_column1]
    subgroup2 = row[group_column2]
    
    subgroup_data = data_frame[(data_frame[group_column1] == subgroup1) & (data_frame[group_column2] == subgroup2)]
    
    min_value = np.nanmin(subgroup_data[value_column])
    max_value = np.nanmax(subgroup_data[value_column])
    
    return min_value, max_value


def adjust_trend(df, columns_to_adjust:list, start_date=None, end_date=None):
    '''
    start and end date must be entered in the format: YYYY-MM-DD
    if dates are not imputed, the function assumes the data is already properly filtered 
    '''
    dataset = df.copy()
    if start_date is not None and end_date is not None:
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset[(dataset['date'] > start_date) & (dataset["date"] < end_date)]

    elif start_date is not None:
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset[dataset['date'] > start_date]

    elif end_date is not None:
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset[dataset['date'] < start_date]
        

    for i in columns_to_adjust:
        dataset[['Min', 'Max']] = dataset.apply(get_min_max, axis=1, args=(dataset, 'keyword', 'country', i), result_type='expand')
        column_name = str(i) + "_adjusted"
        # Adjust values of 'trends_index' between 0 and 100
        dataset[column_name] = (dataset[i] - dataset["Min"]) * 100 / (dataset["Max"] - dataset["Min"])
        dataset = dataset.drop(['Min', 'Max'], axis=1)

    return dataset


def update_column_names(data_list, original_lang_code, wide:bool, df):
    
    mapping_dict = dict(zip(df[original_lang_code], df['en']))

    new_data_list = []
    for data in data_list:
        new_data = data.copy()  # Make a copy of the DataFrame
        new_columns = []

        if wide==True:

            for column in new_data.columns:
                if column in mapping_dict:
                    new_column = f"{mapping_dict[column]}_{original_lang_code}"
                else:
                    new_column = column
                new_columns.append(new_column)
            new_data.columns = new_columns
        
        else:

            for column in new_data.columns:
                if column in mapping_dict:
                    new_column = f"{mapping_dict[column]}"
                else:
                    new_column = column
                new_columns.append(new_column)
            new_data.columns = new_columns


        new_data_list.append(new_data)  # Add the modified DataFrame to the new list


    return new_data_list