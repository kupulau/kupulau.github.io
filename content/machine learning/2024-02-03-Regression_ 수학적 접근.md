---
title: "Regression: 수학적 접근"
category: machine learning
tags:
  - ML
  - statistics
url: https://velog.io/@kupulau/Regression-수학적-접근
created_at: 2024-02-03
related_notes:
---

지도 학습 머신 러닝 문제는 크게 분류 (Classification) 와 회귀 (Regression) 로 나뉜다. 
그 중 회귀에 대해서 알아보겠다.

#### Regression
회귀는 어떤 연속형 데이터 b와 b의 원인으로 추정되는 a 사이의 관계식을 만들어 두 데이터 사이의 관계를 추정하는 방법이다.
→  $b = f(a) + e$     
(여기서 $e$는 오차를 보정하기 위한 수식이다.)
위 식에서 $f$에 어떤 형태의 함수를 사용하느냐에 따라 linear regression, polynomial regression, logistic regression 등으로 나눌 수 있다. 

#### Regularization
머신 러닝에서는 모델이 학습 데이터에 overfit되는 것에 주의해야 한다. 회귀 분석에서 overfit 문제를 해결하는 방법 중 하나는 관계식에 regularization 항을 추가하는 것이다. 이에 대한 자세한 개념적인 정리는 [여기](https://velog.io/@kupulau/Lasso-Regression-Ridge-Regression)에서 볼 수 있다. 

- **L1 regularization** → LASSO(= **L**east **A**bsolute **S**hrinkage and **S**election **O**perator) regression
L1 regularization은 아래와 같은 cost function을 취해서 overfit을 완화하는 것이다. 
>$cost function = \displaystyle\sum_{i=0}^{N} L(y_{true}, y_{model}) +\lambda\sum_{j=0}^M |w_{j}|$ where $w_{j}$ = regression 식에서의 가중치  

&emsp;&emsp;&emsp;L1이라는 이름이 붙은 이유는, regularization 항의 가중치를 더하는 방식이 벡터의 크기를 정의하는 L1 norm의 형태 ||$\overrightarrow{x}$|| =  $\displaystyle\sum_{i=0}^{N}|x_{i}|$ 와 같기 때문이다.


- **L2 regularization**
L2 regularization은 아래와 같은 cost function을 취해서 overfit을 완화하는 것이다.
>$cost function = \displaystyle\sum_{i=0}^{N} L(y_{true}, y_{model}) + \lambda \sum_{j=0}^{M} w_{j}^2$ where $w_{j}$= regression 식에서의 가중치

&emsp;&emsp;&emsp;&emsp;L2라는 이름이 붙은 이유는, regularization 항의 가중치를 더하는 방식이 벡터의 크기를 정의하는 L2 norm의 형태 ||$\overrightarrow{x}$|| = $\displaystyle\sum_{i=0}^{N}x_{i}^2$ 와 같기 때문이다.


두 경우 모두 $\lambda$ 값이 작을수록, regularization의 정도가 약해진다.