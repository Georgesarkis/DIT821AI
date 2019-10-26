
# # House price prediction with Ames Housing Dataset 

import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#### Step 1: Load train_data using Panda's read.csv()
PATH = "./train.csv"
train_house_data = pd.read_csv(PATH)

## Drops the Id column
#train_house_data = train_house_data.drop('Id', axis=1)

## DataFrame.shape returns a tuple representing dimentionsality of the DataFrame
#print('Data shape after removing Id:', train_house_data.shape)  

## DataFrame.info() Prints a summary of a DataFrame
# ---Your code -------     

## DataFrame.head() returns the first five rows
# ---Your code -------

## DataFrame.describe() Generates descriptive statistics 
# ---Your code -------

## DataFrame[column].value_counts() returns values in specified column and their counts
# ---Your code ------- 





#### Step 2 Explore and visualize train data
## Histogram of numerical features
# train_house_data.hist(figsize=(30,20))
#plt.title('Histogram of Numeric features')
#plt.show()

##  Looking for correlations
## Assign the corrections (DataFrame.corr())  to a new object 'feature_corr'
# ---Your code ------- 

# Print correlation of each numeric feature with target 'SalePrice' feature
#print(feature_corr['SalePrice'].sort_values(ascending=False))

## Overview with Scatterplots
# Scatter Matrix
# ---Your code ------- 

# Scatterplot of SalesPrice and GrLivArea
# ---Your code ------- 

## Removing houses with GrLivArea of more than 4000, as these contain outliers
#train_house_data = train_house_data[train_house_data.GrLivArea < 4000]
#print(train_house_data.shape)


## Pivot table to investigate relationship of OveralQual and SalePrice
#quality_pivot = train_house_data.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
#quality_pivot.plot(kind='bar',label='Overall Quality',color='blue')
#plt.ylabel('Median Sale Price')
#plt.show()

# Pivot table to evaluate relationship of SaleCondition and SalePrice
# ---Your code ------- 

# Pivot table to evaluate relationship of Neighborhood and SalePrice
# ---Your code ------- 

# Pivot table to evaluate relationship of MSZoning and SalePrice
# ---Your code ------- 





#Step 3.  Prepare Dataset for Machine Learning Algorithm

## 3.0 Separating features and target variable 
#train_data_features = train_house_data.drop('SalePrice', axis =1) # drop() creates a copy and does not affect original data 
#train_data_target = train_house_data["SalePrice"].copy() 
#train_data_target.columns = ['SalePrice']

# Print DataFrame shape of features and target variable
#print('Shape of features:', train_data_features.shape)
#print('Shape of target:', train_data_target.shape)


#### Handling missing values 
# Check missing values using isnull() .sum() and .sort_values(ascending=False)
#missing_values_train = train_data_features.isnull().sum().sort_values(ascending= False)
#print(missing_values_train.head(30))

# Print DataFrame shape of features and target variable
#print('Expected shape of features :', train_data_features.shape)
#print('Expected shape of target :', train_data_target.shape)


# # Numeric Features (Feature Engineering)
# Get numeric features
# ---Your code ------- 

#print numeric_features with missing values
# ---Your code ------- 


# ## Handling missing values of numeric features
# ---Your code ------- 

# Handle LotFrontage by filling median use DataFrame.groupby() and DataFrame.transform
# ---Your code ------- 


# Get numeric features and check there are no more missing values
#numeric_features = train_data_features.select_dtypes(include=[np.number])
#numeric_features.isnull().sum().sort_values(ascending= False)


# ## Processing of numeric features (log-transform)
# Check skewness of the target variable
#print("Skew (no log transform):", train_house_data.SalePrice.skew())

# Log-transform for target feature 'SalePrice' using numpy
# ---Your code ------- 


# Check skewness of the target variable after log-transform
#print("Skew (log transform):", train_data_target.skew())

# # Categorical Features (Feature Engineering)
# Transforming some numerical variables that are really categorical
# train_data_features['OverallQual'] = train_data_features['OverallQual'].apply(str)
# ---Your code ------- 
# ---Your code ------- 


# Get categorical features
# ---Your code ------- 

# Print categorical_features
# ---Your code ------- 


# Handle missing of categorical features by filling None
# ---Your code ------- 


# Get categorical features and check missing values
# ---Your code ------- 

# ## Processing of categorical features (one-hot encoding)
# Performing one-hot encoding on all categorical features
# ---Your code ------- 

# Print DataFrame shape of features and target variable
# print('Expected shape of features after categorical FE:', train_data_features.shape)
# print('Expected shape of target after categorical FE:', train_data_target.shape)


# # Step 4. Build and Evaluate a simple linear model

# Split dataset usuing `train_test_split()`
# ---Your code ------- 

# Train a simple linear regression model
# ---Your code ------- 

# Make predictions using the trained model on test set
# ---Your code ------- 

# Evaluate model on test set using sklearn's metrics.mean_squared_error()
# ---Your code ------- 

# Evaluate model on train set using sklearn's metrics.mean_squared_error()
# ---Your code ------- 

#print("RMSE on Train set :", RSME_train)
#print("RMSE on Test set :", RSME_test)

# Plotting linear model




