import pandas as pd
import datetime
from time import time
import numpy as np
from math import sqrt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

#Load in training data
df = pd.read_csv('Project Week Data.csv')

# Clean training data to usable format (remove NA rows, columns)
df = df.iloc[:, :-3]
df = df.dropna(axis = 0)

# Convert columns to usable data types (datetimes, ints/floats, dummy variables for categorical)
df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'])
df['DAYOFWEEK'] = pd.to_datetime(df['SALES_DATE']).dt.dayofweek
df['YEAR'] = pd.to_datetime(df['SALES_DATE']).dt.year
df['MONTH'] = pd.to_datetime(df['SALES_DATE']).dt.month
df['DAY'] = pd.to_datetime(df['SALES_DATE']).dt.day
df['DAILY_UNITS'] = df['DAILY_UNITS'].str.replace(',', '').astype(int)
df['RETAIL_PRICE'] = df['RETAIL_PRICE'].str.replace(',', '').astype(float)
df['PROMO'] = np.where(df['PROMO_PRICE'] == "?", 0, 1)
df['Out of Stock'] = np.where(df['Inventory']=='Out-of-Stock', 1, 0)

# Drop unused columns
df = df.drop(columns = ['PROMO_PRICE', 'COMPETITOR_PRICE', 'Inventory'])

# Separate SKUs into DataFrames, all stored in dictionary with SKU as key
grouped = df.groupby(df['Encoded_SKU_ID'])
dfdict = {}
for i in range(1, int(max(df['Encoded_SKU_ID']))+1):
    dfdict[i] = grouped.get_group(i)
    
# Within each DataFrame, drop unused columns, change index to sales date, sort by date
for i in range(1,576):
    dfdict[i] = dfdict[i].drop(columns=['Encoded_SKU_ID'])
    dfdict[i] = dfdict[i].drop(columns=['SUBCLASS_NAME', 'CLASS_NAME', 'ML_NAME', 'CATEGORY_NAME'])
    dfdict[i].index = dfdict[i]['SALES_DATE']
    dfdict[i] = dfdict[i].drop(columns = ['SALES_DATE'])
    dfdict[i] = dfdict[i].sort_index()
    
# Load in validation data
validate = pd.read_csv('Validation_Data.csv')

# Clean validation data to usable format (remove NA rows, columns)
validate = validate.iloc[:, :-12]
validate = validate.dropna(axis = 0)

# Convert columns to usable data types (datetimes, ints/floats, dummy variables for categorical)
validate['SALES_DATE'] = pd.to_datetime(validate['SALES_DATE'])
validate['DAYOFWEEK'] = pd.to_datetime(validate['SALES_DATE']).dt.dayofweek
validate['YEAR'] = pd.to_datetime(validate['SALES_DATE']).dt.year
validate['MONTH'] = pd.to_datetime(validate['SALES_DATE']).dt.month
validate['DAY'] = pd.to_datetime(validate['SALES_DATE']).dt.day
validate['RETAIL_PRICE'] = validate['RETAIL_PRICE'].str.replace(',', '').astype(float)
validate['PROMO'] = np.where(validate['PROMO_PRICE'] == "?", 0, 1)
validate['Out of Stock'] = np.where(validate['Inventory']=='Out-of-Stock', 1, 0)

# Drop unused columns
validate = validate.drop(columns = ['PROMO_PRICE', 'COMPETITOR_PRICE', 'Inventory'])

# Split SKUs into separate DataFrames stored in dictionary (some SKUs not included in validation data)
groupedval = validate.groupby(validate['Encoded_SKU_ID'])
valdict = {}
for i in range(1, 4):
    valdict[i] = groupedval.get_group(i)
for i in range(5, 11):
    valdict[i] = groupedval.get_group(i)
for i in range(12, 27):
    valdict[i] = groupedval.get_group(i)
for i in range(29, 37):
    valdict[i] = groupedval.get_group(i)
for i in range(38, 65):
    valdict[i] = groupedval.get_group(i)
for i in range(66, 70):
    valdict[i] = groupedval.get_group(i)
for i in range(71, 83):
    valdict[i] = groupedval.get_group(i)
for i in range(84, 179):
    valdict[i] = groupedval.get_group(i)
for i in range(180, 196):
    valdict[i] = groupedval.get_group(i)
for i in range(197, 203):
    valdict[i] = groupedval.get_group(i)
for i in range(204, 218):
    valdict[i] = groupedval.get_group(i)
for i in range(219, 220):
    valdict[i] = groupedval.get_group(i)
for i in range(221, 223):
    valdict[i] = groupedval.get_group(i)
for i in range(224, 240):
    valdict[i] = groupedval.get_group(i)
for i in range(241, 255):
    valdict[i] = groupedval.get_group(i)
for i in range(256, 257):
    valdict[i] = groupedval.get_group(i)
for i in range(258, 269):
    valdict[i] = groupedval.get_group(i)
for i in range(270, 285):
    valdict[i] = groupedval.get_group(i)
for i in range(286, 288):
    valdict[i] = groupedval.get_group(i)
for i in range(289, 308):
    valdict[i] = groupedval.get_group(i)
for i in range(309, 332):
    valdict[i] = groupedval.get_group(i)
for i in range(333, 338):
    valdict[i] = groupedval.get_group(i)
for i in range(339, 341):
    valdict[i] = groupedval.get_group(i)
for i in range(342, 355):
    valdict[i] = groupedval.get_group(i)
for i in range(356, 383):
    valdict[i] = groupedval.get_group(i)
for i in range(384, 390):
    valdict[i] = groupedval.get_group(i)
for i in range(391, 394):
    valdict[i] = groupedval.get_group(i)
for i in range(395, 418):
    valdict[i] = groupedval.get_group(i)
for i in range(419, 438):
    valdict[i] = groupedval.get_group(i)
for i in range(440, 451):
    valdict[i] = groupedval.get_group(i)
for i in range(452, 478):
    valdict[i] = groupedval.get_group(i)
for i in range(479, 509):
    valdict[i] = groupedval.get_group(i)
for i in range(510, 546):
    valdict[i] = groupedval.get_group(i)
for i in range(547, 562):
    valdict[i] = groupedval.get_group(i)
for i in range(563, int(max(validate['Encoded_SKU_ID']))+1):
    valdict[i] = groupedval.get_group(i)

# In each DataFrame, drop unused columns, change index to sales date, sort by date
for i in set(valdict.keys()):
    valdict[i] = valdict[i].drop(columns=['Encoded_SKU_ID'])
    valdict[i] = valdict[i].drop(columns=['SUBCLASS_NAME', 'CLASS_NAME', 'ML_NAME', 'CATEGORY_NAME'])
    valdict[i].index = valdict[i]['SALES_DATE']
    valdict[i] = valdict[i].drop(columns = ['SALES_DATE'])
    valdict[i] = valdict[i].sort_index()

# Run ARIMA models for each SKU
for i in set(valdict.keys()):
    # Define exogenous inputs (X), time series data (Y)
    X = dfdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    Y = dfdict[i]['DAILY_UNITS']
    # Create model
    model_sarima = sm.tsa.statespace.SARIMAX(endog = Y, exog = X)
    results = model_sarima.fit()
    # Define exogenous outputs for forecasting
    X_val = valdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    # Create forecast, assign forecasted values to DataFrames
    predict = results.forecast(steps = 7, exog = X_val)
    predict.index = valdict[i].index
    valdict[i]['Forecast'] = predict

# Upon running the initial model using default parameters, we found that some SKUs displayed different patterns, so we took those SKUs and adjusted ARIMA parameters to find better fits.

# Creating set of SKU numbers to reference
patterns = {571, 557, 548, 540, 502, 466, 463, 461, 456, 433, 421, 417, 411, 403, 385, 372, 359, 358, 357, 356, 350, 347, 346, 342, 325, 307, 304, 290, 287, 279, 274, 242, 237, 234, 207, 195, 144, 124, 116, 99, 97, 50, 48}

# Running model among these SKUs (we found p, d, q = 2, 1, 1 as optimal set of parameters)
for i in patterns:
    # Define exogenous inputs (X), time series data (Y)
    X = dfdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    Y = dfdict[i]['DAILY_UNITS']
    # Create model
    model_sarima = sm.tsa.statespace.SARIMAX(order = (2, 1, 1), endog = Y, exog = X)
    results = model_sarima.fit()
    # Define exogenous outputs for forecasting
    X_val = valdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    # Create forecast, assign forecasted values to DataFrames
    predict = results.forecast(steps = 7, exog = X_val)
    predict.index = valdict[i].index
    valdict[i]['Forecast'] = predict

# After running our second model, there were still more with poor fits. We tuned parameters again just for those models.

# Creating set of SKU numbers to reference
others = {242, 271, 463, 502, 313, 24, 555, 274, 342, 548, 347, 35, 200, 483, 515, 385, 403}

# Running model among these SKUs (we found p, d, q = 0, 1, 1 as optimal set of parameters)
for i in others:
    # Define exogenous inputs (X), time series data (Y)
    X = dfdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    Y = dfdict[i]['DAILY_UNITS']
    # Create model
    model_sarima = sm.tsa.statespace.SARIMAX(order = (0, 1, 1), endog = Y, exog = X)
    results = model_sarima.fit()
    # Define exogenous outputs for forecasting
    X_val = valdict[i][['RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock']]
    # Create forecast, assign forecasted values to DataFrames
    predict = results.forecast(steps = 7, exog = X_val)
    predict.index = valdict[i].index
    valdict[i]['Forecast'] = predict

# Adjusting all DataFrames to change negative forecasts to 0, add SKU back to DataFrames
for i in set(valdict.keys()):
    valdict[i]['SKU'] = i
    for j in set(valdict[i].index):
        if valdict[i].loc[j, 'Forecast'] < 0:
            valdict[i].loc[j, 'Forecast'] = 0

# Creating Dictionary of Mean Square Errors
for i in set(valdict.keys()):
    for row in valdict[i]:
        valdict[i]['Square Error'] = (valdict[i]['Forecast'] - valdict[i]['DAILY_UNITS'])**2
MSEs = {i: valdict[i]['Square Error'].mean() for i in set(valdict.keys())}
# Calculating overall Root Mean Square Error (can take mean because all SKUS have same sample size = 7)
from statistics import mean
from math import sqrt
sqrt(mean(MSEs.values()))

# Combine DataFrames into one, output Excel file with forecasts
validation_file = pd.concat(valdict.values())
validation_file = validation_file[['SKU', 'RETAIL_PRICE', 'DAYOFWEEK', 'YEAR', 'MONTH', 'DAY', 'PROMO', 'Out of Stock', 'DAILY_UNITS', 'Forecast']]
validation_file.to_excel('validation.xlsx')

