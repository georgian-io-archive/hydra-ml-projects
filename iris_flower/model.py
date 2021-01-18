import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris

iris = load_iris()

iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_y = pd.DataFrame(iris.target)

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y,
                                                    test_size=0.33, random_state=4)


reg.fit(x_train, y_train)

iris_pred = reg.predict(x_test)

print("Mean squared error")
print(mean_squared_error(y_test, iris_pred))

print("Mean absolute error")
print(mean_absolute_error(y_test, iris_pred))

print("R2 Score")
print(r2_score(y_test, iris_pred))

