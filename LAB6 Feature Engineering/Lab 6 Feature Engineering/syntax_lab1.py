import os
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1. load data
def load_data(data_path, file_name):
    data_path = os.path.join(data_path, file_name)
    return pd.read_csv(data_path, header=None)


my_path = "."
train_file = "ex1data1.txt"

# Direct option: pd.read_csv("./lab1/ex1data1.txt", header=None)
train_data = load_data(my_path, train_file)
train_data.columns = ['Population', 'Profit']  # Add column names

# Step 2.  Explore and visualize data
print(train_data.head(5))
print(train_data.shape)
print(train_data.describe())

# values.reshape() because this array is required to be two-dimensional
X = train_data.Population.values.reshape(-1, 1)
y = train_data.Profit.values.reshape(-1, 1)

print(X)
print(y)

# Plotting the data
plt.scatter(X, y, c='blue')
plt.title('Scatterplot of training data')
plt.xlabel('Population')
plt.ylabel('Profit in $10,000')
plt.show()

# Step 3. Train linear regression mode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
model = lr.fit(X, y)

# Theta as attributes of model using intercept_ for theta_zero and coef_ for theta_one
print('Intercept: ', model.intercept_)
print('Slope: ', model.coef_)

# Step 4. Evaluate trained model
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# Plotting linear model
plt.scatter(X_test, y_test, color='blue')

plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# Evaluate with mean squared error to evaluate the model
print('Mean squared error: ', metrics.mean_squared_error(y_test, y_pred))

# Step 5. Make new predictions
predictions = pd.DataFrame(np.array([[3.5], [7]]), columns=['Population'])

print("Predictions are ", model.predict(predictions))
