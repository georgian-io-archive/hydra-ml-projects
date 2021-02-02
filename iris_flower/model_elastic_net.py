import os
from io import StringIO
import boto3
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from datetime import datetime

## using sklearn
iris = load_iris()

# username = os.environ.get('MLFLOW_TRACKING_USERNAME')
# password = os.environ.get('MLFLOW_TRACKING_PASSWORD')

# TRACKING_URI = f'http://{username}:{password}@mlflow.gcp.impact.georgian.io'

alpha = float(os.environ.get('alpha', 1.0))
max_iter = int(os.environ.get('max_iter', 1000))

mlflow.set_experiment(experiment_name='iris_flower_experiment')
with mlflow.start_run(run_name=f'run-{datetime.now().strftime("%Y%m%d%H%M%S")}') as run:

    en = linear_model.ElasticNet(alpha=alpha, max_iter=max_iter)

    x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

    model = en.fit(x_train, y_train)

    iris_pred = en.predict(x_test)
#
    mlflow.log_param('dataset', 'iris_flower')
    mlflow.log_param('algorithm', 'elastic_net')
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('run_id', run.info.run_id)

    mlflow.log_metric('mean_squared_error', mean_squared_error(y_test, iris_pred))
    mlflow.log_metric('r2_score', r2_score(y_test, iris_pred))

    mlflow.sklearn.log_model(model, "iris_model")

    mlflow.end_run()
