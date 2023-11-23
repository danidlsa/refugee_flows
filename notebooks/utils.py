# this is a python file that defines classes/functions to import the data.
import pandas as pd
import numpy as np

path = '../../data/clean/'

# def load_migration_rates_xlsx():
#     df = pd.read_excel('../../data/WPP2022_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT_REV1.xlsx', skiprows=16)
#     df = df[~df['ISO3 Alpha-code'].isna()].rename({'Net Number of Migrants (thousands)':'n_migrants','Net Migration Rate (per 1,000 population)':'migration_rate', 'ISO3 Alpha-code':'iso' },axis=1)[['Year','iso','n_migrants','migration_rate']]
#     df = df[df.Year >= 2000]
#     return df

# def load_migration_rates():
#     '''This data is annual statistics for each country about:
#         - Net Number of Migrants (thousands)
#         - Net Migration Rate (per 1,000 population)
#         It was taken from https://population.un.org/wpp/Download/Standard/MostUsed/'''
#     df = pd.read_csv(path + 'migration_rates.csv')
#     return df

# def load_migration_stocks_source():
#     '''this data comes from this: https://ourworldindata.org/migration#explore-data-on-where-people-migrate-from-and-to'''
#     df = pd.read_csv('../../data/migration-flows.csv', engine='pyarrow').rename({'year0':'year'},axis=1)

#     df['year'] = pd.to_datetime(df.year, format='%Y')
#     df = df.pivot(columns='dest', index=['year','orig'], values='flow')
#     df.reset_index(level=1, inplace=True)
#     df = df.groupby('orig').apply(lambda x: x.resample('A').ffill())

#     return df  

def load_migration_stocks():
    '''this data is Migration Stocks that comes from Our World in Data: https://ourworldindata.org/migration#explore-data-on-where-people-migrate-from-and-to
    It has been cleaned, namely we have linearly interpolated the data that is collected every 5 years into yearly data, forward filling the missing years.
    Another thing to note- there are some missing values- Ex. Western Sahara (ESH)'''
    df = pd.read_csv(path + 'migration_stocks.csv')
    return df 

def load_idp():
    '''This is data on internally displaced people from the IDMC https://www.internal-displacement.org/database/displacement-data.
    It has been converted into the percent of people with respect to the origin and destination countries.'''
    df = pd.read_csv(path + 'idp.csv')
    return df

# def load_conflict_forecasts():
#     '''This is data on internally displaced people from the IDMC https://www.internal-displacement.org/database/displacement-data.
#     It has been converted into the percent of people with respect to the origin and destination countries.'''
#     df = pd.read_csv(path + 'idp.csv')
#     return df