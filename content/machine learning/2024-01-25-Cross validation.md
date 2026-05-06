---
title: Cross validation
category: machine learning
tags:
  - ML
url: https://velog.io/@kupulau/Cross-validation
created_at: 2024-01-25
related_notes:
---
머신 러닝 중 feature engineering까지 끝냈으면, cross validation이라는 단계를 거치게 된다. cross validation은 training set을 이용해서 모델이 잘 학습되었는지 검증하는 과정이다. cross validation에는 Holdout cross validation, K-fold cross validation, Stratified cross validation 등이 있다.

#### Holdout cross validation
Holdout cross validation은 실전에서 쓰이는 기법이라기보다는, cross validation의 기본적인 개념이라고 보면 되겠다. training set을 일정 비율로 다시 training set과 validation set으로 나눈다. 이 때 training set이 validation set보다 더 많도록 구성한다. (ex) training set : validation set = 8 : 2) 그리고 training set으로 학습시킨 후 이를 validation set을 이용해 잘 학습되었는지 평가한다. 그러나 이와 같이 고정된 validation set을 사용하면 validation set에만 overfit이 일어날 수 있다.

#### K-fold cross validation
Holdout cross validation이 비율을 설정해서 하나의 고정된 validation set을 만드는 것과 달리, K-fold cross validation은 k개의 서로 다른 validation set을 만들어 k번 검증하는 방법이다.
우선 training set을 k개로 분할한다. 이 때 분할된 데이터를 fold라고 한다. k개의 fold들 중 하나를 validation set으로 선택하고, 나머지 k-1 개의 fold를 training set으로 학습을 시키고 검증을 한다. 다음 번에는 이전에 validation set으로 선택되지 않은 fold를 validation set으로 선택하고, 나머지 fold들을 training set으로 하여 학습을 시키고 검증하는 것을 k번 반복한다. 그렇게 총 k개의 검증(예측)값이 나오면, 그들의 평균값으로 모델의 성능을 검증한다.
이같은 방식 덕분에 overfitting 문제가 개선되지만, 시간이 보다 오래 걸린다는 단점도 있다. 

scikit-learn에서 K-fold cross validation 기능을 제공하고 있다.

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier     # KNN model을 예시로 사용해보자.

# KNN model 정의
knn = KNeighborsClassifier(n_neighbors = 10)    

# k_fold로 데이터를 10분할
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) 

# k-fold cross validation -> 10개의 검증(예측)값이 score이라는 리스트에 저장
# x_train = fitting할 데이터 (array)
# y_train = 예측할 데이터 (array)
score = cross_val_score(knn, x_train, y_train, cv=k_fold, n_jobs=-1, scoring='accuracy')
```
KFold와 cross_val_score의 parameter에 대해 더 알아보면 아래와 같다.

- KFold(n_splits, shuffle, random_state)
   - n_splits : 데이터를 분할할 갯수 (default = 5)
   - shuffle : 데이터를 섞은 후 validation set을 선택할지 여부. True/False
   - random_state : shuffle=True일 때, 임의의 정수값을 입력하면 그를 바탕으로 일정하게 랜덤값을 형성해주는 parameter.
   
→ K-fold cross-validation generator를 리턴한다.

- cross_val_score(estimator, x, y, cv, n_jobs, scoring)
   - estimator : 학습할 모델
   - x : fitting할 데이터 (array)
   - y : 예측할 데이터 (array)
   - cv : cross-validation generator
   - n_jobs : 병렬 사용할 CPU 코어의 갯수. n_jobs = -1이면 모든 CPU 코어 사용
   - scoring : 모델 평가 계산식. 보통 scoring = 'accuracy'. 다른 계산식을 쓰고 싶다면 reference 2 참고. 일단 scoring = 'accuracy'로 설정하면 나오는 수치가 1일수록 정확히 예측된 것이다.
   
→ k개의 검증(예측)값을 array로 리턴한다.
   

#### Stratified cross validation
데이터가 편향되어 있을 경우 (어떤 순서에 따라 정렬되어 있는 경우 등), K-fold cross validation을 했을 때 validation set에만 특정 값이 몰려있거나 해서 적절히 학습 및 검증이 되지 않을 수 있다. stratified cross validation은 이런 문제를 해결하기 위해, training set에 A와 B라는 subgroup이 있고 구성 비율이 A : B = 4 : 6이라면, 각 fold에도 A : B = 4 : 6의 비율이 유지되도록 분할하는 방식이다. 
stratified cross validation도 scikit-learn에서 그 기능을 제공하고 있다. parameter는 KFold와 같다.

```python
from sklearn.model_selection import StratifiedKFold

Sk_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
```

#### References
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
https://scikit-learn.org/stable/modules/model_evaluation.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html