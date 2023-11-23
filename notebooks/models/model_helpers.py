# Helper functions - Modelling stage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycountry
import pycountry_convert as pc


# Naive model

def apply_naive_prediction(train, test, target, lst_countries, country_var):
    y_pred = []  # Initialize y_pred as an empty list
    for c in lst_countries:
        train_c = train[train[country_var]==c]
        test_c = test[test[country_var]==c]

        y_pred_1 = train_c[target].iloc[-1]
        y_pred_2 = test_c[target].shift(1)  # Shift the values by 1 timestep
        y_pred_c = y_pred_2.fillna(y_pred_1)
        y_pred.append(y_pred_c)  # Append the predicted values to y_pred list

    return pd.concat(y_pred)  # Concatenate the predicted values into a single series or dataframe


# Multishift function with forward fill
def multi_shift_ffill(data, shift_cols, shift_range, country_var, year_var):
    shifted_data = [data.groupby(country_var)[shift_cols].shift(shift_value) for shift_value in range(shift_range.start, shift_range.stop)]
    shifted_df = pd.concat(shifted_data, axis=1, keys=[f'Shift_{shift_value}' for shift_value in range(shift_range.start, shift_range.stop)])
    an_index = data[[country_var, year_var]].copy()
    shifted_df.columns = shifted_df.columns.map(lambda col: '_'.join(col).strip())
    shifted_df = pd.concat([an_index, shifted_df], axis=1)
    tofill = shifted_df.groupby(country_var).first()
    shifted_df_filled = shifted_df.groupby(country_var).apply(lambda group: group.fillna(tofill.loc[group.name]))
    shifted_df_filled.reset_index(drop=True, inplace=True)
    
    return shifted_df_filled


# Rolling sums
def generate_rolling_sum_variables(data, group_cols, value_cols, window_sizes, date_col):
    panel_data = data.copy()
    panel_data = panel_data.sort_values(by=group_cols + [date_col])
    
    rolling_sums = [
        panel_data.groupby(group_cols)[value_col].transform(lambda x: x.rolling(window, min_periods=1).sum())
        .rename(f'rolling_sum_past_{window-1}_{value_col}')
        for value_col in value_cols
        for window in window_sizes
    ]
    
    panel_data = panel_data.join(pd.DataFrame(rolling_sums).transpose())
    
    return panel_data


## Continent mapping

def country_to_continent(iso3):
    if iso3 == 'UVK':
        return 'EU'
    elif iso3 =='TLS':
        return 'OC'
    elif iso3 =='WBG':
        return 'AS'

    country_alpha2 = pc.country_alpha3_to_country_alpha2(iso3)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    return country_continent_code

def mapper(series, converter, **kwargs):
    unique_keys = series.drop_duplicates()
    unique_vals = unique_keys.apply(converter, **kwargs)
    mapper_dict = dict(zip(unique_keys, unique_vals))
    series = series.map(mapper_dict)
    series.name = series.name + '_continent'
    return series


## Test/Train split, 
def train_test_split(data, target_col, test_time_start, test_time_end, date_var):
    train = data.loc[data[date_var] < test_time_start]
    test = data.loc[(data[date_var] >= test_time_start) & (data[date_var] <= test_time_end)]
    
    X_train = train.drop(columns=target_col)
    y_train = train[target_col]
    
    X_test = test.drop(columns=target_col)
    y_test = test[target_col]
    
    return X_train, X_test, y_train, y_test

# Rolling window
def train_test_split_rw(data, target_col, start_year, end_year, date_var):
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    for year in range(start_year, end_year + 1):
        
        X_train_year = data.loc[data[date_var] < year]
        y_train_year = X_train_year[target_col]
        X_train_list.append(X_train_year.drop(columns=target_col))
        y_train_list.append(y_train_year)
        
        X_test_year = data.loc[data[date_var] == year]
        y_test_year = X_test_year[target_col]
        X_test_list.append(X_test_year.drop(columns=target_col))
        y_test_list.append(y_test_year)
    
    return X_train_list, X_test_list, y_train_list, y_test_list

# Feature importance - graph with 20 main features
def feature_imp_more(feature_importances):
    imp = np.array(list(feature_importances.values()))
    names = list(feature_importances.keys())

    indexes = np.argsort(imp)[-21:]
    indexes = list(indexes)

    plt.barh(range(len(indexes)), imp[indexes], align='center')
    plt.yticks(range(len(indexes)), [names[i] for i in indexes])
    plt.show()

    return indexes


# Define a function to convert ISO2 to ISO3
def convert_iso2_to_iso3(iso2_code):
    try:
        country = pycountry.countries.get(alpha_2=iso2_code)
        return country.alpha_3
    except AttributeError:
        # ISO2 code not found
        return 'N/A'


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


## COSINE SIMILARITY FUNCTION

def cos_similarity(keywords_list, similarity_threshold):
        
    # Create a CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the list of words
    word_vectors = vectorizer.fit_transform(keywords_list)

    # Calculate the cosine similarity matrix
    cosine_similarities = cosine_similarity(word_vectors)

    # Create an empty list to store the results
    results = []

    # Iterate over the words and their cosine similarities
    for i, word in enumerate(keywords_list):
        for j, other_word in enumerate(keywords_list):
            if i != j:
                similarity = cosine_similarities[i, j]
                results.append([word, other_word, similarity])

    # Create a DataFrame from the results list
    df_similarity = pd.DataFrame(results, columns=['Word 1', 'Word 2', 'Cosine Similarity'])

    # Create an empty list to store the column pairs and groups
   
    column_groups = []

    # Iterate over the rows in df_similarity
    for _, row in df_similarity.iterrows():
        word1 = row['Word 1']
        word2 = row['Word 2']
        similarity = row['Cosine Similarity']
        
        if similarity > similarity_threshold:
            found = False
            for group in column_groups:
                if word1 in group or word2 in group:
                    group.add(word1)
                    group.add(word2)
                    found = True
                    break
            if not found:
                column_groups.append({word1, word2})

    # Combine overlapping groups
    merged_groups = []
    for group in column_groups:
        merged = False
        for merged_group in merged_groups:
            if len(group.intersection(merged_group)) > 0:
                merged_group.update(group)
                merged = True
                break
        if not merged:
            merged_groups.append(group)

           
    # Convert groups to list
    column_groups = [list(group) for group in merged_groups]

    return column_groups

def degrees_of_separation(graph, source_name, target_name):
    """
    Returns the number of degrees of separation between two nodes in a graph.
    
    Parameters:
    graph (igraph.Graph): The graph to compute the degrees of separation in.
    source_name (str): The name of the source node.
    target_name (str): The name of the target node.
    
    Returns:
    int: The number of degrees of separation between the source and target nodes.
          Returns None if the nodes are not connected.
    """

    source_idx = graph.vs.find(name=source_name).index
    target_idx = graph.vs.find(name=target_name).index
    shortest_path = graph.distances(source_idx,target_idx)[0][0]
    if shortest_path == np.Inf:
        return -1
    else:
        return shortest_path


class LogExpModelWrapper:
    def __init__(self, model, transform=True):
        self.model = model
        self.transform = transform
        
    def fit(self, X, y):
        transformed_y = np.log1p(y) if self.transform else y
        self.model.fit(X, transformed_y)
        
    def predict(self, X):
        transformed_predictions = self.model.predict(X)
        predictions = np.expm1(transformed_predictions) if self.transform else transformed_predictions
        return predictions
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


def is_spike(series, aggressive=True, threshold=100):
    """
    Determines if a spike is present in the given series.

    Args:
        series (pd.Series): The input series to check for spikes.
        aggressive (bool, optional): If True, uses aggressive spike detection criteria. Defaults to False.
        threshold (float, optional): The threshold value to consider as a spike. Defaults to 100.

    Returns:
        bool: True if a spike is present, False otherwise.
    """

    max_val_index = np.argmax(series)
    max_val = np.max(series)

    # Check for spike conditions based on aggressive mode
    if (max_val == threshold) and (max_val_index == 0) and (series.iloc[max_val_index : max_val_index + 2].sum() == threshold):
        return True
    elif (max_val == threshold) and (max_val_index > 0):
        if aggressive:
            if (series.iloc[max_val_index - 1 : max_val_index + 1].sum() == threshold) or (series.iloc[max_val_index + 1 : max_val_index + 3].sum() == 0):
                return True
        else:
            if series.iloc[max_val_index - 1 : max_val_index + 2].sum() == threshold:
                return True
    
    return False

def smooth_spikes(series, aggressive=True, threshold=100, max_iter=10):
    """
    Smooths out spikes in the given series.

    Args:
        series (pd.Series): The input series to smooth.
        aggressive (bool, optional): If True, uses aggressive spike detection criteria. Defaults to False.
        threshold (float, optional): The threshold value to consider as a spike. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations for spike smoothing. Defaults to 10.

    Returns:
        pd.Series: The smoothed series with spikes removed.
    """

    series = series.copy()
    an_iter = 0

    # Iterate until no more spikes or maximum iterations reached
    while is_spike(series, aggressive, threshold) and (an_iter <= max_iter):
        max_val_index = np.argmax(series)
        next_largest_val = np.partition(series, -2)[-2]

        # Calculate the scaling factor to maintain the overall magnitude
        if next_largest_val == 0:
            scale_factor = 0
        else:
            scale_factor = 100 / next_largest_val

        # Replace the spike value with the average of neighboring values
        if max_val_index < 2:
            series.iloc[max_val_index] = series.iloc[max_val_index : max_val_index + 3].mean()
        else:
            series.iloc[max_val_index] = series.iloc[max_val_index - 2 : max_val_index + 3].mean()

        # Scale the series to maintain the overall magnitude
        series *= scale_factor
        an_iter += 1

    return series

# this does the same thing except reduces the spike even more
def smooth_spikes_2(series, aggressive=True, threshold=100, max_iter=10):
    """
    Smooths out spikes in the given series.

    Args:
        series (pd.Series): The input series to smooth.
        aggressive (bool, optional): If True, uses aggressive spike detection criteria. Defaults to False.
        threshold (float, optional): The threshold value to consider as a spike. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations for spike smoothing. Defaults to 10.

    Returns:
        pd.Series: The smoothed series with spikes removed.
    """

    series = series.copy()
    an_iter = 0

    # Iterate until no more spikes or maximum iterations reached
    while is_spike(series, aggressive, threshold) and (an_iter <= max_iter):
        max_val_index = np.argmax(series)
        next_largest_val = np.partition(series, -2)[-2]

        # Calculate the scaling factor to maintain the overall magnitude
        if next_largest_val == 0:
            scale_factor = 0
        else:
            scale_factor = 100 / next_largest_val

        # Replace the spike value with the average of neighboring values, times .1
        if max_val_index < 2:
            series.iloc[max_val_index] = series.iloc[max_val_index : max_val_index + 3].mean()*.1
        else:
            series.iloc[max_val_index] = series.iloc[max_val_index - 2 : max_val_index + 3].mean()*.1

        # Scale the series to maintain the overall magnitude
        series *= scale_factor
        an_iter += 1

    return series


# Function to convert country name to ISO-3 code
def name_to_iso3(country):
    try:
        country_obj = pycountry.countries.search_fuzzy(country)[0]
        return country_obj.alpha_3
    except LookupError:
        return ''

### PIPELINE FOR TREE MODELS

from category_encoders import BinaryEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def pipeline_tree(model, binary_cols, df):
    be = BinaryEncoder()
    cont_scaler = RobustScaler()

    numerical_cols = list(set(df.columns) - set(binary_cols  + ['year', 'target']))

    from sklearn.compose import ColumnTransformer

    transform_cols = ColumnTransformer(
        [
            ('cat1', be, binary_cols),
    #     ('cat2', ohe, ohe_cols),
            ('num', cont_scaler, numerical_cols)
        ],
        remainder='passthrough'
    )

    pipe = Pipeline([('preprocessing', transform_cols),
                        ('rf', model)])
    return pipe


## Feature importance by year

def feature_years(feature_importances_dict):
        # List of years
    years = [2018, 2019, 2020, 2021]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten the axes to iterate over them easily
    axes = axes.flatten()

    # Iterate over the years and plot the bar plots in the grid
    for i, year in enumerate(years):
        ax = axes[i]
        ax.set_title(f"Year {year}")  # Set title for each subplot with the corresponding year
        
        
        # Sort the feature importances in descending order and get the top 20 features
        top_features = sorted(feature_importances_dict[year].items(), key=lambda x: x[1], reverse=True)[:20]
        top_features = list(reversed(top_features))
        top_features_names, top_features_importances = zip(*top_features)
        
        # Plot the bar chart
        ax.barh(range(len(top_features_names)), top_features_importances, align='center')
        ax.set_yticks(range(len(top_features_names)))
        ax.set_yticklabels(top_features_names)
        ax.set_xlabel('Importance')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plots
    plt.show()
