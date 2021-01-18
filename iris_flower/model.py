import os
import boto3
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris

# using local data
# FILE_PATH = "./data/iris.csv"
# iris_data = pd.read_table(FILE_PATH, sep=",")

# using S3
aws_id = os.environ['AWS_ID']
aws_secret = os.environ['AWS_SECRET']

client = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_acess_key=aws_secret)
bucket = 'gp-sayon-test'
file_path = 'datasets/iris.csv'
csv_object = client.get_object(Bucket=bucket, Key=file_path)

iris_data = pd.read_table(csv_object, sep=",")

iris_x = iris_data.loc[:, 'sepal_length':'petal_width']
iris_y = iris_data.loc[:, 'species':'species']

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y,
                                                    test_size=0.33, random_state=4)

reg.fit(x_train, y_train)

iris_pred = reg.predict(x_test)

print("Mean squared error:", mean_squared_error(y_test, iris_pred))
print("Mean absolute error:", mean_absolute_error(y_test, iris_pred))
print("R2 Score:", r2_score(y_test, iris_pred))

