---
title: Feature importance
category: machine learning
tags:
  - ML
  - feature_engineering
url: https://velog.io/@kupulau/feature-importance
created_at: 2024-01-26
related_notes:
---
여러 feature engineering 작업을 거치다보면 (특히 one-hot encoding) feature의 수가 많이 늘어날 수 있다. 따라서 상대적으로 더 중요한 feature를 구분해내는 작업이 필요하다. 
feature가 머신 러닝에 기여하는 정도를 feature importance (특성 중요도) 라고 하고, 결과에 유의미한 영향을 주는, feature importance가 높은 feature만 중심으로 머신 러닝 기법을 적용하면 더 효율적이다. 
feature importance를 계산하기 전에, 먼저 학습시키려는 데이터에 좋은 성능을 보이는 머신 러닝 기법들부터 추려내 보자. 

```python
import numpy as np

# for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# various models
from sklearn.neighbors import KNeighborsClassifier             
from sklearn.linear_model import LogisticRegression          
from sklearn.svm import SVC                                           
from sklearn.tree import DecisionTreeClassifier       
from sklearn.ensemble import RandomForestClassifier      
from sklearn.ensemble import ExtraTreesClassifier        
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.naive_bayes import GaussianNB                   
from xgboost import XGBClassifier                              
from lightgbm import LGBMClassifier  

knn_model = KNeighborsClassifier()
logreg_model = LogisticRegression()
svc_model = SVC()
decision_model = DecisionTreeClassifier()
random_model = RandomForestClassifier()
extra_model = ExtraTreesClassifier()
gbm_model = GradientBoostingClassifier()
nb_model = GaussianNB()
xgb_model = XGBClassifier(eval_metric='logloss')
lgbm_model = LGBMClassifier()

models =  [knn_model, logreg_model, svc_model, decision_model, random_model, extra_model, gbm_model, nb_model, xgb_model, lgbm_model]

# x_train = training set
# y_train = training set의 정답 데이터

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
results = dict()
i = 0
while i < len(models):
	models[i].fit(x_train, y_train)
    score = cross_val_score(models[i], x_train, y_train.values.ravel(), cv=k_fold, scoring='accuracy')
    results[models[i].__class__.__name__] = np.mean(score)
	i += 1 
```

위와 같은 과정을 거치면, results 딕셔너리에 각 모델의 성능 평가 점수가 기록된다. 1에 가까울수록 잘 예측되었다고 볼 수 있다. 이 점수들을 확인해서 성능이 좋은 머신 러닝 기법만을 이용해서 feature importance를 확인해보자. 예를 들어 Logistic Regression 기법의 점수가 가장 높아서 이의 feature importance를 확인하려면,

```python
logreg_model.feature_importances_

```
와 같이 확인할 수 있다.
이를 성능이 가장 높은 모델 여러 개에 대해 확인해보고, 각 feature importance 값을 모델들에 대해 평균을 내서, 가장 importance가 높은 feature들을 몇 개로 추려낸다. 
