---
title: EfficientDet
category: deep learning
tags:
  - CV
url: https://velog.io/@kupulau/EfficientDet
created_at: 2024-10-06
related_notes:
---

모델을 깊게 쌓을수록 성능이 향상되기는 하지만, 어느 시점부터는 더 쌓는다고 해서 성능이 계속 향상되지 않는다. 기존 연구들에서는 모델의 성능을 향상시키기 위해서 모델을 쌓는 시도를 많이 해 왔는데, 이를 scaling 혹은 scale-up이라고 한다. 무엇을 쌓느냐에 따라 채널을 넓히는 width scaling, 구조를 쌓는 depth scaling, resolution을 조절하는 (이미지 크기를 키우는) resolution scaling, 이 모든 것을 복합적으로 쌓는 compound scaling이 있다. 

### Scaling
#### Width scaling
- MobileNet, MnasNet 등 작은 모델에서 주로 사용됨
- 더 넓은 network는 미세한 특징을 잘 잡아내고, 학습도 쉽다.
- 그러나 극단적으로 넓고 얕은 모델은 상위 level의 특징들을 잘 잡지 못함

#### Depth scaling
- 많은 convolutional network에서 쓰임
- 깊을수록 더 풍부하고 복잡한 특징들을 잡아낼 수 있고, 새로운 태스크에 잘 일반화됨
- 그러나 깊은 network는 gradient vanishing 문제가 생길 수 있음

<br>

### EfficientNet
EfficientNet 연구는 네트워크의 width, depth, resolution의 모든 차원에서 균형을 맞추며 scaling하여 accuracy와 efficiency를 향상시켰다.
이를 위해 object function을 다음과 같이 정의했다.

![](https://velog.velcdn.com/images/kupulau/post/b0c71954-cd36-4257-9bca-e228107e88c3/image.png)

> 다음과 같은 조건을 만족할 때 모델의 accuracy를 최대로 하는 depth, width, resolution을 찾는다.
조건 1. model memory $\leq$ target momory
조건 2. model FLOPS $\leq$ target flops

위와 같은 object function 하에서 scaling을 하며 여러 실험을 해본 결과 "width, depth, resolution을 키우면 정확도가 향상되지만, 큰 모델에 대해서는 정확도 향상 정도가 감소"하고, "scaling 과정에서 width, depth, resolution의 균형을 잘 맞추는 것이 중요"하다는 결과를 얻었다.

위 실험 결과를 바탕으로 **compound scaling method**를 도입했는데, 아래와 같은 조건을 지키면서 width, height, resolution을 scaling하며 accuracy를 가장 높이는 방법을 탐색했다. 

![](https://velog.velcdn.com/images/kupulau/post/f92988d8-2a96-4d17-adce-8dfbff6e8d24/image.png)

이 탐색으로 scaling해서 만든 모델들이 EfficientNet-B0, B1, ..., B7인데 B7로 갈수록 더 높은 성능을 보였다. 

<br>

### EfficientDet
EfficientNet은 분류에 사용된 모델이고, 이를 object detection task에 적용시킨 것이 EfficientDet이다. (1 stage detector) backbone, FPN, box prediction head, classification head를 모두 compound scaling하는 방법을 통해 성능과 속도를 동시에 향상시켰다.