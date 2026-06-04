---
title: tensor와 norm
category: deep learning
tags:
  - DL
  - pytorch
  - math
url: https://velog.io/@kupulau/tensor와-norm
created_at: 2024-08-07
related_notes:
---

tensor의 크기는 어떻게 비교할 수 있을까? tensor의 요소 개수가 많다고 그 tensor가 크다고 할 수 있을까?

### 1-D tensor의 norm
- 1-D tensor에서의 norm은 벡터가 원점에서 얼마나 떨어져 있는지를 의미한다. 따라서 **벡터의 길이**를 측정하는 방법으로 사용된다.
- $L_{p}$ norm 수식 표현
> $||x||_{p}$ = ($\sum_{i=1}^{n}$ $|x|^{p}$)$^{1/p}$
- L1 norm
→ 1-D tensor에 포함된 요소의 절대값의 합
→ `torch.norm(a, p=1)` 하면 tensor a의 L1 norm을 구할 수 있다.
- L2 norm
→ 1-D tensor에 포함된 요소의 제곱합의 제곱근. 두 점 사이의 최단 거리를 측정하는 방법과 같다.
→ `torch.norm(a, p=2)` 로 tensor a의 L2 norm을 구할 수 있다.
- L $\infin$ norm (L-infinity norm이라 읽는다.)
→ 1-D tensor에 포함된 요소들의 절대값 중 최대값
→ `torch.norm(a, p=float('inf'))` 로 tensor a의 L $\infin$ norm을 구할 수 있다.

<br>



### References
https://velog.io/@kupulau/norm%EC%9D%98-%EA%B8%B0%ED%95%98%ED%95%99%EC%A0%81-%EC%9D%98%EB%AF%B8