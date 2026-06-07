---
title: Gradient descent
category: deep learning
tags:
  - ML
  - DL
  - algorithm
url: https://velog.io/@kupulau/Gradient-descent
created_at: 2024-08-08
related_notes:
---

경사하강법 (gradient descent) 는 최적화 알고리즘 중의 하나로, 함수의 기울기를 구해 경사의 반대 방향으로 계속 이동시켜 극소값에 이를 때까지 반복한다.

(이 글은 보다 일반화된 서술을 위하여 가중치와 바이어스를 각각 $x$와 $y$로 표시함)

### 알고리즘

**Step 1.** 학습 데이터가 $x_1$, $y_1$ = (a, b), ... 와 같이 주어졌을 때, 각 데이터 포인트에서의 손실 함수를 계산한다. 이 때 손실 함수를 $L (x, y)$라 하자.

**Step 2.** 그러면 경사(기울기) = $\frac{\partial L(x, y)}{\partial x}$ 를 계산할 수 있게 된다.

**Step 3.** 이 기울기를 바탕으로 $x$를 보다 더 적합한 값으로 업데이트하게 된다. 기울기(미분값)와 반대 방향으로 $x$를 감소시키게 된다. 이 때 $\alpha$ = 학습률(learning rate)로, $x$가 업데이트되는 크기를 결정한다. 모델과 데이터에 따라 달라지는 값이기 때문에, 정해진 값이 있다기 보다는 시행착오를 통해 결정하게 된다.
> $x_{i+1}$ = $x_i$ - $\alpha$ $\frac{\partial L(x, y)}{\partial x}$

**Step 4.** 기울기가 0이 되면 극소값에 도달했으므로 최적화된 $x$ 값에 수렴하게 된다.

**Step 5.** $y$ 또한 같은 방식으로 최적의 값을 구한다. 

<br>

### 확률적 경사하강법
기존의 경사하강법은 모든 데이터를 사용하여 $x$와 $y$를 업데이트하기 때문에 계산이 정확하고 안정적인 장점도 있지만, 그만큼 데이터가 많을수록 계산 비용이 높고 비효율적이게 된다.

이러한 문제점을 해결하고자 나온 것이 **확률적 경사하강법** (stochastic gradient descent) 이다.

- 각각의 데이터 포인트마다 오차를 계산하여 $x$, $y$를 업데이트한다. (모든 데이터의 오차를 계산하지는 않음)
- 따라서 local minima에 갇힐 가능성이 적고 global minima에 잘 도달할 수 있다.

<br>

### 미니 배치 경사하강법
그러나 확률적 경사하강법도 노이즈가 많고, 학습 과정이 불안정해지는 문제가 있다. 이를 보완하고자 나온 것이 미니 배치 경사하강법이다. 

**배치(batch)란, 머신 러닝에서 데이터를 처리하는 묶음 단위이다.** 일반적으로 2의 지수승 (16, 32, 64, ...) 개로 데이터를 묶어서 모델에 학습시킨다. 그렇게 나온 결과(가중치 등)들을 평균내어 사용한다. 따라서 노이즈를 줄일 수 있고 미니 배치마다 가중치를 업데이트하여 계산 속도가 빠르고 학습 과정이 안정적일 수 있다. 

 
<br>

### References
https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95#:~:text=%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95(%E5%82%BE%E6%96%9C%E4%B8%8B%E9%99%8D,%EB%95%8C%EA%B9%8C%EC%A7%80%20%EB%B0%98%EB%B3%B5%EC%8B%9C%ED%82%A4%EB%8A%94%20%EA%B2%83%EC%9D%B4%EB%8B%A4.
https://velog.io/@crosstar1228/MLGradient-Descent-%EC%9D%98-%EC%84%B8-%EC%A2%85%EB%A5%98Batch-Stochastic-Mini-Batch