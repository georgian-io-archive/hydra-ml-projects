import os
from io import StringIO
import boto3
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from datetime import datetime

## using sklearn
iris = load_iris()

iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_y = pd.DataFrame(iris.target)

print(iris_x.shape, iris_y.shape)

## using local data
# FILE_PATH = "./data/iris.csv"
# iris_data = pd.read_table(FILE_PATH, sep=",")

## using S3
# client = boto3.client('s3')
# bucket = 'gp-sayon-test'
# file_path = 'datasets/iris.csv'
# csv_object = client.get_object(Bucket=bucket, Key=file_path)
#
# csv_string = csv_object['Body'].read().decode('utf-8')
#
# iris_data = pd.read_csv(StringIO(csv_string))
#
# iris_x = iris_data.loc[:, 'sepal_length':'petal_width']
# iris_y = iris_data.loc[:, 'species':'species']

# TRACKING_URI = 'http://ec2-3-239-186-96.compute-1.amazonaws.com' # map IP Address to route 53 entry
# mlflow.set_tracking_uri(TRACKING_URI)
#
# n_neighbors = int(os.environ.get('n_neighbors', 5))
# weights = os.environ.get('weights', 'uniform')
#
# with mlflow.start_run(run_name=f'run-{datetime.now().strftime("%Y%m%d%H%M%S")}'):
#
#     knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
#
#     x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y,
#                                                         test_size=0.33, random_state=4)
#
#     model = knn.fit(x_train, y_train)
#
#     iris_pred = knn.predict(x_test)
#
#     mlflow.log_param('dataset', 'iris_flower')
#     mlflow.log_param('algorithm', 'k_neighbors')
#     mlflow.log_param('n_neighbors', n_neighbors)
#     mlflow.log_param('weights', weights)
#
#     mlflow.log_metric('mean_squared_error', mean_squared_error(y_test, iris_pred))
#     mlflow.log_metric('mean_absolute_error', mean_absolute_error(y_test, iris_pred))
#     mlflow.log_metric('r2_score', r2_score(y_test, iris_pred))
#
#     mlflow.sklearn.log_model(model, "iris_model")
#
#     mlflow.end_run()