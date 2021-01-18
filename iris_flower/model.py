import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris

FILE_PATH = "./data/iris.csv"
iris_data = pd.read_table(FILE_PATH, sep=",")

iris_x = iris_data[:, 'sepal_length':'petal_width']
iris_y = iris_data[:, 'species':'species']

print(iris_x)
print(iris_y)

# iris = load_iris()
#
# iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
# iris_y = pd.DataFrame(iris.target)
#
# print(iris.data)
# print(iris.target)
# print(iris.feature_names)
# print(iris.target_names)

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

