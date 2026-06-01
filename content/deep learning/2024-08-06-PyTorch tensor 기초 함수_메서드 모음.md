---
title: PyTorch tensor 기초 함수 메서드 모음
category: deep learning
tags:
  - DL
  - pytorch
url: https://velog.io/@kupulau/PyTorch-tensor-기초-메서드-모음
created_at: 2024-08-06
related_notes:
---

- `min()` : tensor의 모든 요소들 중 최소값을 반환
- `max()` : tensor의 모든 요소들 중 최대값을 반환
- `sum()` : tensor의 모든 요소들의 합을 계산하여 반환
- `prod()` : tensor의 모든 요소들의 곱을 계산하여 반환
- `mean()` : tensor의 모든 요소들의 평균을 계산하여 반환
- `var()` : tensor의 모든 요소들의 표본 분산을 계산하여 반환
- `std()` : tensor의 모든 요소들의 표본 표준 편차를 계산하여 반환
- `.dim()` : tensor 차원의 수 확인
- `.size()` or `.shape` : tensor 사이즈 확인 (m행 n열 확인)
- `.numel()` : tensor 내의 요소의 총 개수 확인

<br>

- 값이 0/1 로 초기화된 1-D tensor 생성하기
```python
a = torch.zeros(n)      # n개의 0으로 이루어진 tensor가 만들어짐  
b = torch.ones(n)       # n개의 1로 이루어진 tensor가 만들어짐
```

- 값이 0/1로 초기화된 2-D tensor 생성하기
```python
a = torch.zeros([m, n])    # m x n 크기의 0으로 이루어진 tensor가 만들어짐
b = torch.ones([m, n])     # m x n 크기의 1로 이루어진 tensor가 만들어짐
```

- 값이 0/1로 초기화된 3-D tensor 생성하기
```python
a = torch.zeros([m, n, l])   # m x n x l 크기의 0으로 이루어진 tensor가 만들어짐
b = torch.one([m, n, l])     # m x n x l 크기의 1로 이루어진 tensor가 만들어짐
```

- 기존의 tensor를 크기와 자료형이 같지만 값이 0/1로 초기화된 tensor로 변환하기
```python
a = torch.zeros([1,1,1,1], dtype=int8)

b = torch.zeros_like(a)
c = torch.ones_like(a)
```

- [연속균등분포](https://velog.io/@kupulau/%EC%97%B0%EC%86%8D%EA%B7%A0%EB%93%B1%EB%B6%84%ED%8F%AC)에서 추출한 난수로 채워진 tensor 생성
```python
a = torch.rand(n)      # 연속균등분포에서 추출한 n개의 난수를 요소로 가지는 1-D tensor   
b = torch.rand([m, n])    # m x n 크기의 난수를 요소로 가지는 2-D tensor
```

- 표준정규분포에서 추출한 난수로 채워진 tensor 생성
```python
a = torch.randn(n)   # 표준정규분포에서 추출한 n개의 난수를 요소로 가지는 1-D tensor   
b = torch.randn([m, n])  # m x n 크기의 난수를 요소로 가지는 2-D tensor
```

- 기존의 tensor를 크기와 자료형이 같지만 값이 난수로 된 tensor로 변환하기
```python
a = torch.zeros([1,1,1,1], dtype=int8)
b = torch.rand_like(a)    # 연속균등분포에서 추출한 난수로 변환됨
c = torch.randn_like(a)   # 표준정규분포에서 추출한 난수로 변환됨
```

- arange
```python
a = torch.arange(start = 1, end = 9, step = 2)
# resulting tensor contains [1, 3, 5, 7]
```

- 초기화되지 않은 tensor 생성하기
초기화되지 않았다는 것은 tensor의 요소가 명시적으로 특정 값 (0, 1 등) 으로 설정되지 않고, 메모리에 이미 존재하는 임의의 값들로 채워진다는 것을 의미한다. 
```python
a = torch.empty(n)     # n개의 요소로 채워진 1-D tensor 생성
b = torch.empty([m, n])    # m x n 크기의 2-D tensor 생성
```

- 초기화되지 않은 tensor의 요소를 다른 값으로 수정하기
이 기능을 이용하면 메모리 주소가 변경되지 않는다.
```python
a.fill_(x)    # a 텐서의 요소가 x 값로 변경된다.
```

- 리스트를 tensor로 만들기
```python
list1 = [1, 2, 3, 4, 5]
tensor1 = torch.tensor(list1)
```

- numpy array를 tensor로 만들기
```python
import numpy as np

np1 = np.array([[1, 2], [3, 4]])
tensor1 = torch.from_numpy(np1)
```
**numpy array를 tensor로 변환할 때 주의할 점!**
numpy array로 만든 tensor는 정수형이기 때문에 실수 데이터를 다루고 있다면 실수형으로 type casting 해주어야 한다.
```python
tensor2 = torch.from_numpy(np1).float()
```

- [CPU Tensor](https://velog.io/@kupulau/CPU-tensor-GPU-tensor) 생성하기
```python
a = torch.IntTensor([1, 2, 3, 4])    # 정수형
b = torch.FloatTensor()    # 실수형
c = torch.ByteTensor()    # 8비트 부호 없는 정수형
d = torch.CharTensor()    # 8비트 부호 있는 정수형
e = torch.ShortTensor()   # 16비트 부호 있는 정수형
f = torch.LongTensor()    # 64비트 부호 있는 정수형
g = torch.DoubleTensor()  # 64비트 부호 있는 실수형
```

- tensor 복제
```python
x = torch.tensor([1, 2, 3, 4])
y = x.clone()
z = x.detach()
```
Q. clone과 detach 메서드는 무엇이 다른가?

|연산|값 복사|메모리 공유 (원본에 영향)|연산 그래프 유지|역전파 가능|
|:--:|:--:|:--:|:--:|:--:|
|`clone()`|deep copy|❌|✅|✅|
|`detach()`|shallow copy|✅|❌|❌|

`clone`은 텐서의 값을 복사하여 새로운 텐서를 생성하지만, 기존 연산 기록(그래디언트 계산 그래프)은 유지하는 상태로 텐서를 복사하는 기능이다. 따라서 역전파 계산을 하고 싶을 때 사용한다.

`detach`는 텐서의 연산 그래프를 끊고, autograd(자동 미분)를 비활성화한 상태로 텐서를 복사하는 기능이다. 따라서 gradient 계산에서 제외할 때 사용한다. 


- CUDA tensor 생성
```python
a = torch.tensor([1, 2, 3, 4]).to('cuda')
a = torch.tensor([1, 2, 3, 4]).cuda()
```

- GPU에 할당된 tensor를 CPU tensor로 변환하기
```python
b = a.to(device='cpu')
b = a.cpu()
```
