import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_boston

import os

boston = load_boston()

boston_x = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_y = pd.DataFrame(boston.target)

alpha = int(os.environ.get('alpha', 1.0))
max_iter = int(os.environ.get('max_iter', 1000))

reg = linear_model.ElasticNet(alpha=alpha, max_iter=max_iter)

print(f'alpha={alpha}, max_iter={max_iter}')

x_train, x_test, y_train, y_test = train_test_split(boston_x, boston_y,
                                                    test_size=0.33, random_state=4)


reg.fit(x_train, y_train)

boston_pred = reg.predict(x_test)

print("Mean squared error")
print(mean_squared_error(y_test, boston_pred))

print("Mean absolute error")
print(mean_absolute_error(y_test, boston_pred))

print("R2 Score")
print(r2_score(y_test, boston_pred))