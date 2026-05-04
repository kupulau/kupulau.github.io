---
title: Bias-Variance tradeoff
category: machine learning
tags:
  - ML
  - statistics
url: https://velog.io/@kupulau/Bias-Variance-tradeoff
created_at: 2024-01-21
related_notes:
---

#### Accuracy and Precision

![](https://velog.velcdn.com/images/kupulau/post/0b2b38a6-ff85-40de-a458-fd829c640caf/image.png)

bias-variance tradeoff를 이해하기 위해 먼저 accuracy와 precision의 개념을 짚고 가야 한다.
accuracy(정확도)는 측정값/예측값이 실제값과 얼만큼 가까운지를 나타내는 개념이고, precision(정밀도)은 여러 번 측정한 값들이 서로 얼마나 가까운지를 나타내는 개념이다. 위의 그림에서 accuracy와 precision의 의미를 직관적으로 확인할 수 있다. 

#### Bias-Variance tradeoff
관측값/예측값이 accurate한지, precise한지는 관측값/예측값의 bias와 variance를 통해 판단할 수 있다. bias는 관측값/예측값들이 특정 방향으로 치우친 정도를 나타내는 값이고, variance는 여러 값들이 분산되어 있는 정도를 나타내는 값이다. 위의 그림으로 보면, 좌측 상단의 accurate, but not precise가 low bias, high variance인 경우이고, 우측 하단의 precise, but not accurate가 high bias, low variance인 경우이다.

#### 머신 러닝에서의 적용
![](https://velog.velcdn.com/images/kupulau/post/f17e63fc-0c2c-4a81-a51c-8a581abecc74/image.png)

실제 머신 러닝에서는 training set의 예측값을 통해 bias를 파악하고, test set의 예측값을 통해 variance를 파악하면 된다. 
bias가 클 때에는 보다 복잡한 모델을 사용하거나 fitting을 더 최적화 시킬 수 있도록 하고, variance가 클 때에는 데이터 수를 늘리거나 feature 수를 줄이는 등의 시도를 해보는 것이 좋다. 

<br>

#### References
https://en.wikipedia.org/wiki/Bias–variance_tradeoff
https://en.wikipedia.org/wiki/Accuracy_and_precision
https://wp.stolaf.edu/it/gis-precision-accuracy/