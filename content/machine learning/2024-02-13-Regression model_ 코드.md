---
title: "Regression model: 코드"
category: machine learning
tags:
  - ML
  - python
url: https://velog.io/@kupulau/Regression-model-코드
created_at: 2024-02-13
related_notes:
---

#### 1. Linear Regression
```python
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()

# x_train = training set
# y_train = training set의 정답 데이터
# x_test = test set

LR_train_model = LR_model.fit(x_train, y_train)
LR_predicted = LR_train_model.predict(x_test)
```

#### 2. Lasso Regression
```python
from sklearn.linear_model import Lasso

lasso_model = Lasso()

lasso_train_model = lasso_model.fit(x_train, y_train)
lasso_predicted = lasso_train_model.predict(x_test)
```

#### 3. Ridge Regression
```python
from sklearn.linear_model import Ridge

ridge_model = Ridge()

ridge_train_model = ridge_model.fit(x_train, y_train)
ridge_predicted = ridge_train_model.predict(x_test)
```

#### 4. Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()

RF_train_model = RF_model.fit(x_train, y_train)
RF_predicted = RF_train_model.fit(x_test)
```

#### 5. XGBoost Regressor

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor()

xgb_train_model = xgb_model.fit(x_train, y_train)
xgb_predicted = xgb_train_model.fit(x_test)
```

#### 6. Gradient Boosting Regressor
```python
from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()

GB_train_model = GB_model.fit(x_train, y_train)
GB_predicted = GB_train_model.fit(x_test)
```

** 좀 더 정확한 결과를 위해 GridSearch, RandomSearch 등 hyperparameter tuning을 해서 parameter 값을 모델에 입력해주면 좋다.

