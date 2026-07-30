---
title: Kullback-Leibler divergence
category: statistics
tags:
  - statistics
url: https://velog.io/@kupulau/Kullback-Leibler-divergence
created_at: 2024-12-07
related_notes:
---

Kullback-Liebler divergence(KL divergence)는 두 확률 분포 P와 Q 간의 차이를 측정하는 척도이다. 일반적으로 P는 실제 분포, Q는 근사 분포 (모델이 학습하려는 분포)로 간주된다.

수학적으로 KL divergence는 아래와 같이 정의된다.

>$D_{KL}(P||Q) = \sum_xP (x) \, log \frac{P(x)}{Q(x)}$ (discrete)
>
> $D_{KL}(P||Q) = \int P (x) \, log \frac{P(x)}{Q(x)} \, dx$ (continuous)

$D_{KL}$ 값이 0에 가까울수록 두 분포가 동일한 것이고, 값이 커질수록 두 분포가 다르다는 것을 의미한다. 
