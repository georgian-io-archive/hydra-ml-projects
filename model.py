from datetime import datetime
import os

import mlflow.xgboost
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

eta = float(os.environ.get('eta', 0.2))
max_depth = int(os.environ.get('max_depth', 5))
subsample = float(os.environ.get('subsample', 1))
lambda_param = float(os.environ.get('lambda', 1))
alpha = float(os.environ.get('alpha', 0))
min_child_weight = int(os.environ.get('min_child_weight', 1))


mlflow.set_experiment(experiment_name='breast_cancer_experiment')
with mlflow.start_run(run_name=f'run-{datetime.now().strftime("%Y%m%d%H%M%S")}') as run:

    # if more than one evaluation metric are given the last one is used for early stopping
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                  random_state=42,
                                  eval_metric="auc",
                                  eta=eta,
                                  max_depth=max_depth,
                                  subsample=subsample,
                                  reg_lambda=lambda_param,
                                  reg_alpha=alpha,
                                  min_child_weight=min_child_weight)

    mlflow.log_param('eta', eta)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('subsample', subsample)
    mlflow.log_param('lambda', lambda_param)
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('min_child_weight', min_child_weight)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

    y_pred = xgb_model.predict(X_test)

    mlflow.log_metric('accuracy_score',
                      accuracy_score(y_test, y_pred))

    mlflow.xgboost.log_model(xgb_model, "cancer_model")

    mlflow.end_run()
