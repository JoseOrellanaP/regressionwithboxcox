import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(72018)

def to_2s(array):
    return array.reshape(array.shape[0], -1)

def plot_exponencial_data():
    data = np.exp(np.random.normal(size=1000))
    plt.hist(data)
    plt.show()
    return data

def plot_square_normal_data():
    data = np.square(np.random.normal(loc=5, size=1000))
    plt.hist(data)
    plt.show()
    return data


# Loading the Bostom Housind data

file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_data = pd.read_csv(file_name)


boston_data.head()


# =============================================================================
# Determining normally distribuited target
# If our target is not normally distribuited we can apply a transformation 
# to it and then fit out regression to predict the transformed values

# There are two ways to tell if our target is normally transform
# - Using a visual approach
# - Using a statistical test

# =============================================================================

# Using a visual approach
# Plotting a histogram

boston_data.MEDV.hist()


# Using a statistical test

## The test output a p-value. The higher this p-value is the closer the distributions is to normal
## Frequentist statisticians would say that you accept that the distribution is normal
## (fail to reject the null hyphothesis that it is normal) if p > 0.05

from scipy.stats.mstats import normaltest

normaltest(boston_data.MEDV.values) # P value is extremely low. The y variable is not normally distribuited


# Apply transformation to make target variable more normally distribuited for regression
## Linear Regression assumes the residuals to be normally distribuited which can be aided by
## making y (the target variable) normally distribuited. There are some ways to get so:   
 ## Log transformation
 ## Square root transformation
 ## Box cox trasnformation
 

################## Log transformation

## The log transformation can transform data that is significant skewed rigth to be more normally distribuited

data = plot_exponencial_data()
plt.hist(data)
plt.hist(np.log(data))

# Apply transformation to Bostom Housing data

log_medv = np.log(boston_data.MEDV)
plt.hist(log_medv)
normaltest(log_medv) # It's close to normal distribution but it still isn't


################## Square root transformation

data = plot_square_normal_data()
plt.hist(np.sqrt(data))


sqrt_medv = np.sqrt(boston_data.MEDV)
plt.hist(sqrt_medv)
normaltest(sqrt_medv)



################## Box cox transformation

from scipy.stats import boxcox

bc_result = boxcox(boston_data.MEDV)
boxcox_medv = bc_result[0]
lam = bc_result[1]

boston_data.MEDV.hist()
plt.hist(boxcox_medv)
normaltest(boxcox_medv)





# Testing regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)


lr = LinearRegression()


# Define and load the predictior (x) and target (y) variables

y_col = "MEDV"

x = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]


# Create a polynomial features

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)

# Splitting the data into training and testing set

x_train, x_test, y_train, y_test = train_test_split(x_pf, y, test_size=.3, random_state=72018)

# Normalize the data using StandarScale on x_train.

s = StandardScaler()
x_train_s = s.fit_transform(x_train)

bc_result2 = boxcox(y_train)
y_train_bc = bc_result2[0]
lam2 = bc_result2[1]



y_train_bc.shape
lr.fit(x_train_s, y_train_bc)
x_test_s = s.transform(x_test)
y_pred_bc = lr.predict(x_test_s)


# Apply inverse transformation to be able to use it in regression context

from scipy.special import inv_boxcox

inv_boxcox(boxcox_medv, lam)[:10]

boston_data["MEDV"].values[:10]



y_pred_tran = inv_boxcox(y_pred_bc, lam2)
r2_score(y_test, y_pred_tran)


# Determining s core without boxcox transformation

lr2 = LinearRegression()
lr2.fit(x_train_s, y_train)
y_pred2 = lr2.predict(x_test_s)
r2_score(y_test, y_pred2)

























































