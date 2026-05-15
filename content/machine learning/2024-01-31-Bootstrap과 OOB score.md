---
title: Bootstrap과 OOB score
category: machine learning
tags:
  - ML
  - statistics
url: https://velog.io/@kupulau/Bootstrap과-OOB-score
created_at: 2024-01-31
related_notes:
---
#### Bootstrapping
![](https://velog.velcdn.com/images/kupulau/post/532d84c8-f6c1-4878-a8cc-38bc4df72bfa/image.png)
부트스트랩은 원래의 샘플에서 중복을 허용하여 무작위로 임의의 sample set을 여러 개 만들고 이를 통해 분포, 신뢰구간 등의 통계값을 계산하여 모집단의 분포 등을 추정하는 기법이다.

중복을 허용하여 데이터를 추출하기 때문에, 사용되지 않는 데이터가 생길 수 있는데 평균적으로 원래 샘플의 1/3 정도가 사용되지 않는다고 한다. 이 사용되지 않은 데이터를 **O**ut-**O**f-**B**ag (OOB) 샘플이라고 한다. 

#### OOB score
OOB 샘플은 학습에는 사용되지 않았기 때문에, test set으로써 역할을 할 수 있다. 따라서 학습 데이터로 나온 예측값과 OOB 샘플로 테스트한 결과값의 오차를 기반으로 정확도를 계산할 수 있다. 이를 OOB score이라고 하고, 이를 기반으로 머신 러닝의 성능 평가를 할 수 있다. 