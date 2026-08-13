---
title: 모델 경량화 기법 (3) Quantization
category: deep learning
tags:
  - AI
url: https://velog.io/@kupulau/모델-경량화-기법-3-Quantization
created_at: 2025-01-05
related_notes:
---

Quantization(양자화)은 모델의 가중치와 활성화를 낮은 비트 정밀도로 변환하여 저장 및 계산 효율성을 높이는 기법이다. 간단한 예를 들어, $\pi$의 값을 3.141592로 표현하면 정밀도는 높지만 메모리를 많이 차지하게 되고, 3으로 표현하면 정밀도는 낮지만 메모리를 적게 사용할 수 있기 때문에 후자의 표현으로 경량화를 하는 것이다. 하지만 계산 속도 향상과 메모리 효율성만을 생각하다가 모델의 성능이 크게 감소하면 곤란하기 때문에, 오차를 최소화하는 적절하게 낮은 정밀도를 찾는 것이 중요하다.

## Quantization Mapping

![](https://velog.velcdn.com/images/kupulau/post/9d917f93-e077-49f5-a694-e8482fe5cc6b/image.png)

quantization mapping은 높은 정밀도의 값을 적절한 낮은 정밀도 값에 매핑하는 방법이다. 이러한 매핑은 데이터 전체 단위로 진행할 수도 있고, 레이어 단위로 파라미터 매핑을 할 수도 있다.

한편, 대부분의 데이터, 모델 파라미터 등은 자료형이 FP32이므로, FP16 혹은 INT8 등으로 오차를 최소화하면서 변환하여 사용하는 방법이 있다. 그러나 비트 수에 따라서 표현 가능한 범위가 달라지는 것에 유의해야 한다.

|타입|표현 가능 최소|표현 가능 최대|
|:--:|:--:|:--:|
|INT8|-128|127|
|INT16|-32768|32767|
|INT32|−214748364|2147483647|
|FP16|−65504|65504|
|FP32|−3.4028235 $\times$ 10$^{38}$|3.4028235 $\times$ 10$^{38}$|

예를 들어, FP32로 표현된 350.5는 INT8로 표현하는 것이 불가능할 것이다. 왜냐하면 INT8의 표현 가능한 최대 범위는 127이기 때문이다.

### 방법
Quantization mapping은 양자화된 값을 저장하고 이를 복원(de-quantization)하는 방식으로 진행된다. 이 때 양자화 방식 (메타데이터), 양자화된 값, scale factor $s$ (기울기 값), zero-point $z$ (0의 양자화 후 위치) 값을 저장하여 사용한다.

$X_{quant} = round(\frac{X}{s}+z)$

$X_{dequant} = s \times (X_{quant} - z)$

($X$ = 원본 / $X_{quant}$ = 양자화 된 값 / $X_{dequant}$ = 복원된 값)

Quantization mapping의 방법은 크게 absmax quantization과 zero-point Quantization이 있다.

#### Absmax Quantization
absmax quantization은 변환 후 값의 범위가 0을 기준으로 좌우 대칭이 되도록 변환하는 것이다. 데이터 분포가 대칭적이거나 평균 값이 0인 경우에 적용하며, activation function이 tanh일 때 효과적이다. ($\because$ tanh 함수는 -1 ~ 1 사이의 대칭적인 분포이기 때문) 그러나 극단적인 값 (outlier)에 민감하다는 점에 주의해야 한다.

absmax quantization에서 scale factor와 zero-point는 아래와 같이 계산한다. 

$s = \frac{max|X|}{127}$

$z = 0$

absmax quantization의 경우 $z$가 항상 0이기 때문에 따로 저장하지 않는다.


#### Zero-point Quantization
zero-point quantization은 0을 고려하지 않고 전체 범위를 균일하게 변환하는 것이다. 데이터 분포가 비대칭적이거나 평균 값이 0이 아닌 경우에 적용하며, activation function이 ReLU일 때 효과적이다. ($\because$ y = 0 when x < 0 and y > 0 when x > 0 for ReLU) 그러나 기준점이 비정상적일 경우에는 성능이 떨어질 수 있다. 

zero-point quantization에서 scale factor는 데이터에서 최댓값이 127, 최솟값이 -128이 되도록 계산한다.

$s = \frac{max X - min X}{255}$

$z = -128 - round(\frac{min X}{s})$


### Clipping
앞서 absmax quantization에서 살펴봤듯 데이터에 outlier가 포함되어 있다면 효과적인 quantization이 어렵다. 이 문제를 해결하기 위해 값이 일정 범주를 넘어가는 경우 같은 값으로 취급하는 clipping이라는 방법을 이용한다. 
예를 들어 범주를 \[-5, 5]로 지정하는데 256이라는 값이 있다면 이를 5로 바꾼 후에 양자화 하는 것이다. 이 때 적절한 범주(range)를 찾는 과정을 calibration이라고 한다.