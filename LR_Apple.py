from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('apple.csv')

y = dataset.iloc[:, -3:-2].values
X = np.arange(628).reshape((-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/10, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.plot(X, y, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('variable x')
plt.ylabel('variable y')
plt.show()
