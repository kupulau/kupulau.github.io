---
title: Nearest neighbor classifier
category: machine learning
tags:
  - ML
  - algorithm
url: https://velog.io/@kupulau/Nearest-neighbor-classifier
created_at: 2024-08-12
related_notes:
  - "[[2024-01-18-머신 러닝 기법]]"
---

Nearest neighbor classifier는 데이터 분류 알고리즘의 한 종류이다. 자세한 설명은 머신러닝 기법의 [[2024-01-18-머신 러닝 기법#K-Nearest Neighbor (KNN)]] 참고.

이미지 분류하는 작업을 nearest neighbor로 한다고 가정하자. 해당 이미지가 어떤 종류인지 분석을 할 때, 판단하려는 이미지와 기준 이미지가 얼마나 차이나는지를 계산해야 할 것이다. 컴퓨터가 인식하기에 이미지는 픽셀값을 가지는 2D matrix이므로, 두 matrix 사이의 거리 메트릭을 활용하면 되겠다. 이를 활용해서 Nearest neighbor classifier를 클래스로 구현하면 아래와 같다.

```python
import numpy as np

class NearestNeighbor: 
	def __init__(self):
		pass
        
	def train(self, images, labels)     # 이미지와 라벨로 학습
    	self.images = images
		self.labels = labels

	def predict(self, test_image):    # 이미지가 1D vector라고 가정
		min_dist = sys.maxint  
        for i in range(self.images.shape[0]):
        	dist = np.sum(np.abs(self.images[i, :] - test_image))
    		if dist < min_dist:
      			min_dist = dist
      			min_index = i
		return self.labels[min_index]
```

그러나 Nearest neighbor는
- 학습을 할 때의 시간 복잡도가 O(1), 예측을 할 때의 시간 복잡도가 O(N)으로 비효율적이고 느리며,
- 픽셀의 거리 메트릭은 이미지에 대해 의미있는 정보를 제공하지 않으며,
- 고차원 데이터가 될수록 필요한 예제의 수가 기하급수적으로 늘어나기 때문에 (차원의 저주)

이미지 분류라는 작업에서는 거의 사용되지 않는 알고리즘이다.