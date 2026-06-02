# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data: Read the CSV file containing student scores, extract the independent variable (X) and dependent variable (Y), and split the data into training and testing sets.

2.Train Model: Use the training data to train a linear regression model using LinearRegression() from scikit-learn.

3.Predict Test Set Results: Predict the target variable Y for the test set using the trained model.

4.Visualize Results: Plot the training data with the regression line, and similarly plot the test data with the predicted line.

5.Evaluate Model: Calculate and display performance metrics (Mean Squared Error, Mean Absolute Error, and Root Mean Squared Error).

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHINESHKUMAR E
RegisterNumber:212224220026
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
print(Y_pred)

print(Y_test)

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
<img width="871" height="573" alt="363060378-2e450af3-e206-4b55-b7e4-ca87b65a9a68" src="https://github.com/user-attachments/assets/f3533f5f-a28f-4c4f-b129-08c639084aeb" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
