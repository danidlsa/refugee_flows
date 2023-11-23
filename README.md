# Forecasting Global Refugee Flows: A Machine Learning Approach using Non-conventional Data

## Daniela de los Santos, Eric Frey and Renato Vasallo
### Barcelona School of Economics 
This repository contains the data and code used for the paper 'Forecasting Global Refugee Flows: A Machine Learning Approach using Non-conventional Data'. Most of the research was conducted in the framework of our master's thesis submission for Data Science Decision Making 2023, in partnership with the United Nations High Commissioner for Refugees (UNHCR). Our goal is to optimize the yearly refugee flows prediction between dyads of countries.

### How to navigate this repository:
+ `data`: The data folder contains all the data sources that were used for this project, with the exception of UNHCR data, which was shared under discretion. It contains a `clean_data.ipynb` file, which is how the `raw` data is processed and subsequently moved to the `clean` folder.
+ `notebooks/models`: This folder contains the main notebooks that implement refugee outflows and flow predictions, along with helper functions files. It is the key part of the repository.
+ `notebooks/trends`: Within this folder all of the Google Trends downloading and pretesting takes place. You will find a notebook that summarizes the workflow for downloading data, along with helper functions files and helper csv files as well. As this was done in different stages, we recommend approaching the folder as a sample of the total of our work. Feel free to consult the authors for specific queries if you need to.
  
### About the project

This study presents a novel forecasting framework for global refugee flows, incorporating non-conventional data sources such as Google Trends, the GDELT project event dataset, conflict forecasts, among others. Our main objective is to generate accurate one-step ahead predictions for the number of new refugee arrivals per country pair. These predictions play a crucial role in facilitating effective humanitarian response and informed infrastructure planning. While existing literature focuses on developed countries, this research develops a comprehensive global model, considering the majority of refugees seeking refuge in neighboring countries. In addition, to overcome challenges with imbalanced and low-frequency data, two strategies are proposed using a rolling window framework: modeling refugee outflows from origin countries (outflow level), and modeling refugee flows between country pairs (dyad level). Our results reveal a significant improvement in prediction accuracy by augmenting traditional variables from UNHCR with high-frequency non-conventional data, with Random Forest and Gradient Boosting as effective regressors.

### Contact the authors:
Daniela De los Santos - Barcelona School of Economics - daniela.de@bse.eu

Eric Frey - Barcelona School of Economics - eric.frey@bse.eu

Renato Vassallo - Barcelona School of Economics renato.vassallo@bse.eu
