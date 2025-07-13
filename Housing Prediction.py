# House Price Prediction using ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt




#link for the dataset Link:- https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

X = df[['rm', 'lstat', 'ptratio']]
# 'rm': average number of rooms per dwelling
# 'lstat': % lower status of the population
# 'ptratio': pupil-teacher ratio by town


y = df['medv']
#medv is teh median value.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


## chatgpt is used to find dataset and resolve minimal errors