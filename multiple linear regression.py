import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing libraries
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 4].values

#Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X = LabelEncoder()
#labelEncoder_X = labelEncoder_X.fit(X[:, 0])
#X[:, 0] = labelEncoder_X.transform(X[:, 0])
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

#splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Avoiding the dummy variable trap
X = X[:,1:]

#importing regressior from sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicing the y_test set
y_pred = regressor.predict(X_test)

#backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int),values = X,axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#splitting dataset(based on opt dataset) into training set and test set
from sklearn.cross_validation import train_test_split
X_opt_train,X_opt_test,Y_opt_train,Y_opt_test = train_test_split(X_opt,Y,test_size = 0.2,random_state = 0)

#removing X0 column from X_opt_train and X_opt_test
X_opt_train = X_opt_train[:, 1:]
X_opt_test = X_opt_test[:, 1:]

#fitting regression to opt_training set
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train,Y_opt_train)

#predicting the test set results
y_opt_pred = regressor_opt.predict(X_opt_test)

#plotting the training set results
plt.scatter(X_opt_train,Y_opt_train,color = 'red')
plt.plot(X_opt_train,regressor_opt.predict(X_opt_train),color = 'blue')
plt.title('Profit vs R&D(training set)')
plt.xlabel('R&D(Amount spent by companies in $)')
plt.ylabel('Profit(in $)')
plt.show()

#plotting the test set results
plt.scatter(X_opt_test,Y_opt_test,color = 'red')
plt.plot(X_opt_train,regressor_opt.predict(X_opt_train),color = 'blue')
plt.title('Profit vs R&D(test set)')
plt.xlabel('R&D(Amount spent by companies in $)')
plt.ylabel('Profit(in $)')

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""