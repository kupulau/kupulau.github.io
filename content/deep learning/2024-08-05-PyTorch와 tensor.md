---
title: PyTorch와 tensor
category: deep learning
tags:
  - DL
  - pytorch
url: https://velog.io/@kupulau/PyTorch와-tensor
created_at: 2024-08-05
related_notes:
---

## PyTorch란
머신러닝 알고리즘을 구현·실행하기 위한 딥러닝 라이브러리이다.
파이썬에서 `import torch` 해서 사용할 수 있다.


## Tensor
tensor는 PyTorch의 기본 데이터 구조이다.

### tensor의 종류
- 0-D tensor (= scalar) : 하나의 숫자로 표현되는 양 ex) 서울의 기온 28.4도
```python
a = torch.tensor(28.4)
```

- 1-D tensor (= 1d array; vector)
```python
a = torch.tensor([72, 9.3, 34, 120, 15.7])
```

- 2-D tensor : 동일한 크기를 가진 1-D tensor들이 모여서 형성한, 행과 열로 구성된 구조 (= 2d array; matrix)
```python
a = torch.tensor([[72, 9.3, 34, 120, 15.7],
				  [3.9, 560, 146, 9, 13],
				  [0.62, 280, 24, 33, 14.7]])
```

- 3-D tensor : 동일한 크기의 2-D tensor들이 쌓여 형성된 입체적인 배열 구조
```python
a = torch.tensor([[[255, 0, 0],
				   [0, 255, 0]],
				  [[0, 0, 255],
				  [255, 255, 0]]])
```
tensor를 쌓을 때에는 stack 메서드를 이용하면 된다.
```python
tensor_a = torch.tensor([[[255, 0, 0],
				   [0, 255, 0]],
				  [[0, 0, 255],
				  [255, 255, 0]]])
                  
tensor_b = torch.tensor([[[255, 0, 0],
				   [0, 255, 0]],
				  [[0, 0, 255],
				  [255, 255, 0]]])
                  
tensor_c = torch.tensor([[[255, 0, 0],
				   [0, 255, 0]],
				  [[0, 0, 255],
				  [255, 255, 0]]])                

# dimension 2 를 기준으로 stack
tensor3d = torch.stack((tensor_a, tensor_b, tensor_c), dim=2) 
```

★ 3-D tensor의 구조 
m x n x l 의 크기를 가지는데, m개의 n x l 행렬이 있는 것으로 해석할 수 있다.
m이 dim = 0, n이 dim = 1, l이 dim = 2

- N-D tensor : 동일한 크기의 (N-1)-D tensor들이 쌓여 형성된 입체적인 배열 구조 (N $\geq$ 4)