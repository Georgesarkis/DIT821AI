
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
train_house_data = train_house_data.drop('Id', axis=1)

## DataFrame.shape returns a tuple representing dimentionsality of the DataFrame
train_house_data.shape

## DataFrame.info() Prints a summary of a DataFrame
train_house_data.info()

## DataFrame.head() returns the first five rows
print(train_house_data.head())

## DataFrame.describe() Generates descriptive statistics
train_house_data.describe()

## DataFrame[column].value_counts() returns values in specified column and their counts
train_house_data["Street"].value_counts()

#### Step 2 Explore and visualize train data
## Histogram of numerical features
train_house_data.hist(figsize=(30,20))
plt.title('Histogram of Numeric features')
plt.show()

##  Looking for correlations
## Assign the corrections (DataFrame.corr())  to a new object 'feature_corr'
feature_corr = train_house_data.corr()

# Print correlation of each numeric feature with target 'SalePrice' feature
print(feature_corr['SalePrice'].sort_values(ascending=False))

## Overview with Scatterplots
# Scatter Matrix
#
list = ["OverallQual","GrLivArea", "GarageArea", "TotalBsmtSF" , "LotArea"]
#for var in list:
#    data = pd.concat([train_house_data['SalePrice'], train_house_data[var]], axis=1)
#    data.plot.scatter(x= var, y='SalePrice', ylim =(0,800000))
#    plt.show()

#iris = train_house_data
iris_df = pd.DataFrame(train_house_data["SalePrice"], columns=train_house_data['SalePrice'])
iris_df[list] = train_house_data[list]
pd.plotting.scatter_matrix(iris_df, alpha=0.9, figsize=(10, 10))
plt.show()

# Scatterplot of SalesPrice and GrLivArea
data = pd.concat([train_house_data['GrLivArea'], train_house_data["SalePrice"]], axis=1)
data.plot.scatter(x="SalePrice", y='GrLivArea')
plt.show()

## Removing houses with GrLivArea of more than 4000, as these contain outliers
train_house_data = train_house_data[train_house_data.GrLivArea < 4000]
print(train_house_data.shape)


## Pivot table to investigate relationship of OveralQual and SalePrice
quality_pivot = train_house_data.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar',label='Overall Quality',color='blue')
plt.ylabel('Median Sale Price')
plt.show()

# Pivot table to evaluate relationship of SaleCondition and SalePrice
quality_pivot = train_house_data.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar',label='Overall Quality',color='blue')
plt.ylabel('Median Sale Price')
plt.show()

# Pivot table to evaluate relationship of Neighborhood and SalePrice
quality_pivot = train_house_data.pivot_table(index='Neighborhood', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar',label='Overall Quality',color='blue')
plt.ylabel('Median Sale Price')
plt.show()

# Pivot table to evaluate relationship of MSZoning and SalePrice
quality_pivot = train_house_data.pivot_table(index='MSZoning', values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar',label='Overall Quality',color='blue')
plt.ylabel('Median Sale Price')
plt.show()




#Step 3.  Prepare Dataset for Machine Learning Algorithm

## 3.0 Separating features and target variable
train_house_data = train_house_data.dropna(subset=["Electrical"])
train_data_features = train_house_data.drop('SalePrice', axis =1)
train_data_target = train_house_data["SalePrice"].copy()
train_data_target.columns = ['SalePrice']

# Print DataFrame shape of features and target variable
print('Shape of features:', train_data_features.shape)
print('Shape of target:', train_data_target.shape)


#### Handling missing values 
# Check missing values using isnull() .sum() and .sort_values(ascending=False)
missing_values_train = train_data_features.isnull().sum().sort_values(ascending= False)
print(missing_values_train.head(30))

# Print DataFrame shape of features and target variable
print('Expected shape of features :', train_data_features.shape)
print('Expected shape of target :', train_data_target.shape)


# # Numeric Features (Feature Engineering)
# Get numeric features
numeric_features = train_data_features.select_dtypes(include=[np.number])
numeric_features.isnull().sum().sort_values(ascending= False)

#print numeric_features with missing values
print(" numeric_features with missing values: ", numeric_features.isnull().sum().sort_values(ascending= False))

# ## Handling missing values of numeric features
train_data_features["GarageYrBlt"].fillna(0, inplace = True)
train_data_features["MasVnrArea"].fillna(0, inplace = True)

# Handle LotFrontage by filling median use DataFrame.groupby() and DataFrame.transform
neighborhood = train_data_features.groupby("Neighborhood")
train_data_features.LotFrontage = neighborhood["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Get numeric features and check there are no more missing values
numeric_features = train_data_features.select_dtypes(include=[np.number])
numeric_features.isnull().sum().sort_values(ascending= False)
print(" numeric_features with missing values: ", numeric_features.isnull().sum().sort_values(ascending= False))

# ## Processing of numeric features (log-transform)
# Check skewness of the target variable
print("Skew (no log transform):", train_house_data.SalePrice.skew())

# Log-transform for target feature 'SalePrice' using numpy
train_data_target = np.log(train_house_data['SalePrice'])

# Check skewness of the target variable after log-transform
print("Skew (log transform):", train_data_target.skew())

# # Categorical Features (Feature Engineering)
# Transforming some numerical variables that are really categorical
train_data_features['OverallQual'] = train_data_features['OverallQual'].apply(str)
train_data_features['MSSubClass']= train_data_features['MSSubClass'].apply(str)
train_data_features['OverallCond'] =train_data_features['OverallCond'].apply(str)
train_data_features['YrSold'] = train_data_features['YrSold'].apply(str)
train_data_features['MoSold'] = train_data_features['MoSold'].apply(str)

# Get categorical features
banana = train_data_features.shape

# Print categorical_features
print('Shape all_data: {}'.format(banana))


# Handle missing of categorical features by filling None
train_data_features["PoolQC"].fillna("None", inplace = True)
train_data_features["MiscFeature"].fillna("None", inplace = True)
train_data_features["Alley"].fillna("None", inplace = True)
train_data_features["Fence"].fillna("None", inplace = True)
train_data_features["FireplaceQu"].fillna("None", inplace = True)
train_data_features["GarageCond"].fillna("None", inplace = True)
train_data_features["GarageType"].fillna("None", inplace = True)
train_data_features["GarageFinish"].fillna("None", inplace = True)
train_data_features["GarageQual"].fillna("None", inplace = True)
train_data_features["BsmtExposure"].fillna("None", inplace = True)
train_data_features["BsmtFinType2"].fillna("None", inplace = True)
train_data_features["BsmtQual"].fillna("None", inplace = True)
train_data_features["BsmtCond"].fillna("None", inplace = True)
train_data_features["BsmtFinType1"].fillna("None", inplace = True)
train_data_features["MasVnrType"].fillna("None", inplace = True)

# Get categorical features and check missing values
numeric_features = train_data_features.select_dtypes(exclude=[np.number])
numeric_features.isnull().sum().sort_values(ascending= False)
print(" numeric_features with missing values: ", numeric_features.isnull().sum().sort_values(ascending= False))

# ## Processing of categorical features (one-hot encoding)
# Performing one-hot encoding on all categorical features
train_data_features = train_data_features.select_dtypes(include=[np.number])
train_data_features = pd.get_dummies(train_data_features)

# Print DataFrame shape of features and target variable
print('Expected shape of features after categorical FE:', train_data_features.shape)
print('Expected shape of target after categorical FE:', train_data_target.shape)


# # Step 4. Build and Evaluate a simple linear model

# Split dataset usuing `train_test_split()`
Xtrain, Xtest, ytrain, ytest = train_test_split(train_data_features, train_data_target, test_size= 0.2, random_state=0)

# Train a simple linear regression model
lr = LinearRegression()
model= lr.fit(Xtrain, ytrain)

# Make predictions using the trained model on test set
ypredtest = model.predict(Xtest)
ypredtrain = model.predict(Xtrain)
# Evaluate model on test set using sklearn's metrics.mean_squared_error()
RSMEtest= metrics.mean_squared_error(ytest, ypredtest)

# Evaluate model on train set using sklearn's metrics.mean_squared_error()
RSMEtrain= metrics.mean_squared_error(ytrain, ypredtrain)

print("RMSE on Train set :", RSMEtrain)
print("RMSE on Test set :", RSMEtest)
plt.show()

# Plotting linear model
plt.scatter(ytest, ypredtest, color='green')
plt.scatter(ytrain, ypredtrain, color='blue')

plt.show()



