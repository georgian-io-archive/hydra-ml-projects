import os
from io import StringIO
import boto3
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn

# using local data
FILE_PATH = "./data/iris.csv"
iris_data = pd.read_table(FILE_PATH, sep=",")

# client = boto3.client('s3')
# bucket = 'gp-sayon-test'
# file_path = 'datasets/iris.csv'
# csv_object = client.get_object(Bucket=bucket, Key=file_path)
#
# csv_string = csv_object['Body'].read().decode('utf-8')
#
# iris_data = pd.read_csv(StringIO(csv_string))

TRACKING_URI = 'http://ec2-3-239-186-96.compute-1.amazonaws.com'
mlflow.set_tracking_uri(TRACKING_URI)

iris_x = iris_data.loc[:, 'sepal_length':'petal_width']
iris_y = iris_data.loc[:, 'species':'species']

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y,
                                                    test_size=0.33, random_state=4)

model = reg.fit(x_train, y_train)

iris_pred = reg.predict(x_test)

with mlflow.start_run():
    mlflow.log_metric('metric1', 2)
    mlflow.log_param('param1', 3)
    # mlflow.sklearn.log_model(model, "iris_model")

print("Mean squared error:", mean_squared_error(y_test, iris_pred))
print("Mean absolute error:", mean_absolute_error(y_test, iris_pred))
print("R2 Score:", r2_score(y_test, iris_pred))

