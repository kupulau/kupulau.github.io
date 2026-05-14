---
title: 여러 모델로 hyperparameter tuning 연습
category: machine learning
tags:
  - ML
  - coding
url: https://velog.io/@kupulau/여러-모델로-hyperparameter-tuning-연습
created_at: 2024-01-29
related_notes:
---
importance가 높은 feature를 추려낸 데이터를 이용해서 여러 모델의 hyperparameter tuning을 연습해보자.
```python
# x_train = training set
# y_train = training set의 정답 데이터
```


#### Support Vector Machine (SVM)
SVM에서 사용자가 결정해야 하는 hyperparameter로는 (최소) C와 gamma가 있다. 
C = regularization parameter. training set에만 overfit되지 않도록 수학식을 추가해서 결과값이 덜 변하도록 하는 파라미터이다. 양수여야 한다.
gamma = 경계선 모양을 변화시키는 정도. 0 ~ 1 정도의 값을 가지는 듯?

```python
from sklearn.svm import SVC     
```

- Grid search
```python
from sklearn.model_selection import GridSearchCV

hyperparams = {'C':[10, 15, 20, 25, 30, 50], 
				'gamma':[0.01, 0.05, 0.06, 0.07, 0.1]}
                
grid = GridSearchCV(estimator = SVC(random_state=1), 
				  param_grid = hyperparams,
				  verbose = True,    # 코드 실행 중간 중간에 state를 알려주는 옵션
                  cv = 5,    # integer 입력시 stratified k-fold 분할할 갯수
                  scoring = 'accuracy',
                  n_jobs = -1)
                  
grid.fit(x_train, y_train)

print(grid.best_score_)     # 가장 높은 accuracy
print(grid.best_params_)    # 가장 높은 accuracy를 낼 때의 C와 gamma 값을 리턴 
```

- Random search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

# stats.uniform(a, b) 이용시 a ~ b 범위에서 균등한 확률로 값 추출
hyperparams = {'C': stats.uniform(0, 50),
    		    'gamma': stats.uniform(0, 1)}

random = RandomizedSearchCV(estimator = SVC(random_state=1),
							param_distributions = hyperparams,
                            n_iter = 100,
                            cv = 5, 
                            scoring = 'accuracy',
                            random_state = 1,
                            n_jobs = -1)
                 
random.fit(x_train, y_train)

print(random.best_score_)    # 가장 높은 accuracy
print(random.best_params_)    # 가장 높은 accuracy를 낼 때의 C와 gamma 값을 리턴
```

<br>

#### Gradient Boosting Machine (GBM)
GBM에서 사용자가 결정해야 하는 hyperparameter로는 (최소) learning rate와 n_estimator, max_depth가 있다. 
learning rate = weak learner가 오류 값을 보정해 나가는 데 적용하는 계수. 0 ~ 1 사이의 값을 가짐. 너무 작은 값을 적용하면 업데이트 되는 값이 작아져서 최소 오류 값을 찾아 예측 성능이 높아질 가능성이 크지만 수행 시간이 오래 걸리고, 너무 큰 값을 적용하면 예측 성능이 떨어질 가능성이 있지만 빠른 수행이 가능하다.
n_estimators = weak learner의 개수
max_depth = (weak learner로 사용하는) 트리의 깊이

- Grid search

```python
from sklearn.ensemble import GradientBoostingClassifier

learning_rate = [0.01, 0.05, 0.1, 0.2]
n_estimators = [100, 1000, 2000]
max_depth = [3, 5, 10, 15]

hyperparams = {'learning_rate' : learning_rate,
			   'n_estimators' : n_estimators,
               'max_depth' : max_depth}
               
grid = GridSearchCV(estimator = GradientBoostingClassifier(random_state=1),
                    param_grid = hyperparams,
                    verbose = True,
                    cv = 5,
                    scoring = 'accuracy',
                    n_jobs = -1)

grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_params_)    # 가장 높은 accuracy를 낼 때의 learning rate, n_estimators, max_depth 값을 리턴
```

<br>

#### Logistic Regression
Logistic regression에서 사용자가 결정해야 하는 hyperparameter로는 (최소) penalty와 C가 있다.
penalty = regularization의 종류. 'l1', 'l2' ,'elasticnet', None 중에 선택한다. 자세한 것은 ([Ridge Regression, Lasso Regression](https://velog.io/@kupulau/Lasso-Regression-Ridge-Regression) 참조)
C = regularization parameter. SVM의 C와 같음. 양수여야하고, 수가 클수록 결과값이 크게 변한다. (맞는지 확인필)

```python
from sklearn.linear_model import LogisticRegression
```

- Grid search
```python
import numpy as np

penalty = ['l1', 'l2']
Clist = np.linspace(700, 900, 200)

hyperparams = {'penalty': penalty, 'C' : Clist}

grid = GridSearchCV(estimator = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000), 
    				param_grid = hyperparams,
                    verbose=True, 
                    cv=5, 
                    scoring = "accuracy", 
                    n_jobs=-1)

grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_params_)    # 가장 높은 accuracy를 낼 때의 penalty, C 값을 리턴
```


- Random search
```python
from scipy.stats import randint

hyperparams = {'penalty': ['l1', 'l2', 'elasticnet'], 
               'C': stats.uniform(0, 1000)}

random = RandomizedSearchCV(estimator = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000), 
    						param_distributions=hyperparams, 
                            n_iter=100, 
                            cv=5, 
                            scoring='accuracy',
                            random_state=1,
                            n_jobs=-1)

random.fit(x_train, y_train)

print(random.best_score_)    # 가장 높은 accuracy
print(random.best_params_)    # 가장 높은 accuracy를 낼 때의 penalty, C 값을 리턴
```

<br>

#### XGBoost
ensemble 기법은 실행시 자동으로 각 파라미터를 적절히 맞춰주기 때문에 hyperparameter tuning이 예측 정확도를 높이는 데 큰 기여를 하는 편은 아니다. 여러 weak learner들로 인해 파라미터가 무수히 많아지고 수행 시간이 오래 걸리기 때문에 중요한 파라미터들을 중심으로 튜닝을 하는 것이 좋다. 

>-  learning rate : 이전 결과를 얼마나 반영할지, 학습 단계별로 적용할 가중치를 의미한다. 일반적으로 0.01 ~ 0.2 사이의 값을 많이 사용한다.
- max_depth : (weak learner로 사용하는) 트리의 최대 깊이. max_depth = -1이면 트리의 최대 깊이에 제한이 없다는 의미이다. 보통 3 ~ 10 사이의 값을 많이 사용한다.
-  gamma : regularization parameter. gamma 값이 클수록 트리에 가지가 적게 만들어진다. 0 이상의 값을 사용한다.
-  min_child_weight : 값이 작을수록 트리에 가지가 더 많이 만들어진다. 0 이상의 값을 사용한다.
-  subsample : 트리를 만들 때 사용되는 training set의 비율. 보통 0.5 ~ 1 사이의 값이 많이 사용된다.
-  colsample_bytree : 트리를 만들때 사용되는 feature (column)의 비율. 0.5 ~ 1 사이의 값을 많이 사용한다.
- reg_alpha : l1 regularization (Lasso regression) 가중치
- reg_lambda : l2 regularization (Ridge regression) 가중치

- Grid Search
위의 많은 파라미터들을 한꺼번에 넣을 경우 수행시간이 매우 길어지므로 여러 단계로 나누어 각 단계에서 일부 파라미터만 튜닝을 하고, 그 다음 단계에서는 튜닝된 파라미터들을 입력하고 나머지 파라미터들을 튜닝하는 식으로 하면 효율적이다. (다만 아래 코드 예시에서는 한꺼번에 하겠다.)

```python
from xgboost import XGBClassifier

learning_rate = [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.12, 0.15, 0.17, 0.2]
n_estimators = [10, 50, 60, 75, 85, 100, 125, 150, 200, 250, 500, 1000]
max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
min_child_weight = [1, 2, 3, 4, 5, 6, 7]
gamma =  [i*0.1 for i in range(0,5)]
subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]

hyperparams = {'learning_rate': learning_rate, 
               'n_estimators': n_estimators,
               'max_depth': max_depth, 
               'min_child_weight': min_child_weight,
               'gamma': gamma,
               'subsample':subsample,
               'colsample_bytree':colsample_bytree,
               'reg_alpha': reg_alpha}

grid=GridSearchCV(estimator = XGBClassifier(random_state=1, eval_metric='logloss'), 
                  param_grid = hyperparams, 
                  verbose=True, 
                  cv=5,
                  scoring = "accuracy", 
                  n_jobs=-1)
                  
grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_parms_)    # 가장 높은 accuracy를 낼 때의 파라미터 값을 리턴
```

- Bayesian Optimization

이제 Bayesian Optimization을 이용해서 XGBoost 모델의 hyperparameter tuning을 해보자. 그 전에 Bayesian Optimization 함수에 대해 잠시 짚고 넘어가보자면
```python
BayesianOptimization(f, pbounds, random_state)
```
>- f : bayesian optimization이 적용될 함수
- pbounds : 함수에 들어가는 parameter들의 범위 (딕셔너리)

```python
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

pbounds = {'learning_rate': (0.01, 0.5),  
    	   'n_estimators': (100, 1000), 
           'max_depth': (3, 10),
           'min_child_weight': (0, 10),
           'subsample': (0.5, 1.0),  
           'colsample_bytree': (0.5, 1.0),   
           'gamma': (0, 5)
           'reg_lambda': (0, 1000, 'log-uniform'),
           'reg_alpha': (0, 1.0, 'log-uniform')}    

def xgboost_hyper_param(learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    xgb = XGBClassifier(max_depth=max_depth, 
    					min_child_weight= min_child_weight,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators, 
                        subsample=subsample, 
                        colsample_bytree=colsample_bytree, 
                        gamma=gamma,
                        random_state=1,
                        eval_metric='logloss'
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda)                  
    return np.mean(cross_val_score(xgb, x_train, y_train, cv=5, scoring='accuracy'))
    
optimizer = BayesianOptimization(f=xgboost_hyper_param, pbounds=pbounds, random_state=1)

optimizer.maximize(init_points=10, n_iter=100, acq='ei', xi=0.01)
# init_points = 초기 랜덤 포인트 갯수
# n_iter = 반복 횟수
# acq = 최적화 값을 찾기 위한 수식 옵션. ei(=Expected Improvement)가 가장 많이 쓰임.
# xi = 최종 수학식에 반영하는 비율

optimizer.max   # 가장 높은 성능일 때의 파라미터 값 리턴
```

<br>

#### LightGBM
LightGBM도 ensemble 기법의 일종이기 때문에 사용자가 튜닝할 하이퍼 파라미터는 XGBoost와 동일하다.

- Bayesian Optimization
```python
from lightgbm import LGBMClassifier 

pbounds = {'learning_rate': (0.01, 0.5),  
           'n_estimators': (100, 1000), 
           'max_depth': (3, 10),
           'min_child_weight': (0, 10),    
           'subsample': (0.5, 1.0),
           'colsample_bytree': (0.5, 1.0)
           'reg_lambda': (0, 1000),
           'reg_alpha': (0, 1.0)}

def lgbm_hyper_param(learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    lgbm = LGBMClassifier(max_depth=max_depth,
                          min_child_weight=min_child_weight,
                          learning_rate=learning_rate, 
                          n_estimators=n_estimators, 
                          subsample=subsample, 
                          colsample_bytree=colsample_bytree,
                          random_state=1
                          reg_lambda=reg_lambda,        
                          reg_alpha=reg_alpha)
    return np.mean(cross_val_score(lgbm, x_train, y_train, cv=5, scoring='accuracy'))  
    
optimizer = BayesianOptimization(f=lgbm_hyper_param, pbounds=pbounds, verbose=1, random_state=1)

optimizer.maximize(init_points=10, n_iter=100, acq='ei', xi=0.01)

optimizer.max  # 가장 높은 성능일 때의 파라미터 값 리턴
```

<br>

#### K-Nearest Neighbor (KNN)
KNN은 k값을 튜닝해주면 되겠다.

- Grid Search
```python
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hyperparams = {'n_neighbors': n_neighbors}

grid = GridSearchCV(estimator = KNeighborsClassifier(), 
                   param_grid = hyperparams, 
                   verbose=True, 
                   cv=5, 
                   scoring = "accuracy", 
                   n_jobs=-1)

grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_parms_)    # 가장 높은 accuracy를 낼 때의 k값을 리턴
```

<br>

#### Random Forest
마찬가지로 ensemble 기법의 일종이기 때문에 사용자가 튜닝할 하이퍼 파라미터는 XGBoost와 동일하다.

- Grid Search

```python
from sklearn.ensemble import RandomForestClassifier 

n_estimators = [10, 50, 100, 200]
max_depth = [3, None] 
max_features = [0.1, 0.2, 0.5, 0.8, 'sqrt', 'log2'] # 트리를 만들 때 feature를 사용하는 비율
min_samples_split = [2, 4, 6, 8, 10] # 노드를 분할하기 위한 최소 샘플의 수
min_samples_leaf = [2, 4, 6, 8, 10] # 리프 노드가 되기 위해 필요한 최소 샘플의 수

hyperparams = {'n_estimators': n_estimators, 
               'max_depth': max_depth, 
               'max_features': max_features,
               'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf}

grid = GridSearchCV(estimator = RandomForestClassifier(random_state=1), 
                    param_grid = hyperparams, 
                    verbose=True, 
                    cv=5, 
                    scoring = "accuracy", 
                    n_jobs=-1)

grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_parms_)    # 가장 높은 accuracy를 낼 때의 파라미터 값을 리턴
```

<br>

#### Extra Tree
마찬가지로 ensemble 기법의 일종이기 때문에 사용자가 튜닝할 하이퍼 파라미터는 XGBoost와 동일하다.

- Grid Search
```python
from sklearn.ensemble import ExtraTreesClassifier

n_estimators = [10, 50, 100, 200]
max_depth = [3, None] 
max_features = [0.1, 0.2, 0.5, 0.8, 'sqrt', 'log2'] 
min_samples_split = [2, 4, 6, 8, 10] 
min_samples_leaf = [2, 4, 6, 8, 10] 

hyperparams = {'n_estimators': n_estimators, 
               'max_depth': max_depth, 
               'max_features': max_features,
               'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf}

grid=GridSearchCV(estimator = ExtraTreesClassifier(random_state=1), 
                  param_grid = hyperparams, 
                  verbose=True, 
                  cv=5, 
                  scoring = "accuracy", 
                  n_jobs=-1)

grid.fit(x_train, y_train)

print(grid.best_score_)    # 가장 높은 accuracy
print(grid.best_parms_)    # 가장 높은 accuracy를 낼 때의 파라미터 값을 리턴
```


<br>

#### References
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://for-my-wealthy-life.tistory.com/37
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
https://velog.io/@sset2323/04-05.-GBMGradient-Boosting-Machine
https://xgboost.readthedocs.io/en/stable/get_started.html
https://github.com/bayesian-optimization/BayesianOptimization
https://www.cognex.com/ko-kr/blogs/deep-learning/research/overview-bayesian-optimization-effective-hyperparameter-search-technique-deep-learning-1