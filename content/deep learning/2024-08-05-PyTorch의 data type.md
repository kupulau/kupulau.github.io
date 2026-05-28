---
title: PyTorch의 data type
category: deep learning
tags:
  - DL
  - pytorch
url: https://velog.io/@kupulau/PyTorch의-data-type
created_at: 2024-08-05
related_notes:
---
PyTorch에서 데이터 타입이라고 하면, 기본 구조인 tensor가 저장하는 값의 dtype을 의미한다. (ex) `int`, `float`)

### 정수형 (integer)
- 8비트 부호 없는 정수 : 8 bit를 사용하여 0 ~ 255의 정수를 표현하는 데이터 타입
`a = torch.tensor(1, dtype = torch.uint8)`


- 8비트 부호 있는 정수 : 8 bit를 사용하여 -128 ~ 127까지의 정수를 표현하는 데이터 타입. 가장 앞의 1 bit를 부호를 나타내는 데 사용한다.
`a = torch.tensor(-1, dtype = torch.int8)`


- 16비트 부호 있는 정수 : 16 bit를 사용하여 -32,768 ~ 32,767까지의 정수를 표현하는 데이터 타입.
`a = torch.tensor(1, dtype = torch.short)` 혹은
`a = torch.tensor(1, dtype = torch.int16)` 로도 표현할 수 있다.


- 32비트 부호 있는 정수 : 32 bit를 사용하여 -2,147,483,648 ~ 2,147,483,647 까지의 정수를 표현하는 데이터 타입. 대부분의 프로그래밍에서표준적인 정수 크기로 사용한다.
`a = torch.tensor(1, dtype = torch.int)` 혹은
`a = torch.tensor(1, dtype = torch.int32)` 로도 표현할 수 있다.
 
- 64비트 부호 있는 정수 : 64 bit를 사용하여 -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807 까지의 정수를 표현하는 데이터 타입.
`a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.long)` 혹은
`a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.int64)` 로도 표현할 수 있다.

<br>

### 실수형 (float)

◎ 실수형 데이터 타입은 신경망의 수치 계산에서 사용된다.
◎ 아래에서 나오는 부동 소수점, 가수부, 지수부에 대한 자세한 설명은 [여기](https://velog.io/@kupulau/고정-소수점과-부동-소수점) 참조.

- 32비트 부동 소수점 수 : 32 bit를 사용하여 가수부와 지수부로 표현하는 데이터 타입
`a = torch.tensor(1, dtype = torch.float32)` 혹은
`a = torch.tensor(1, dtype = torch.float)` 로도 표현할 수 있다.


- 64비트 부동 소수점 수 : 64 bit를 사용하여 지수부와 가수부로 표현하는 데이터 타입
`a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.double)` 혹은
`a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float64)` 로도 표현할 수 있다.


<br>

### 타입 캐스팅 (type casting)

타입 캐스팅이란 어떤 데이터 타입을 다른 데이터 타입으로 변환하는 것을 의미한다.

- 8비트 부호 있는 정수 → 32비트 부동 소수점 수로 변환
```python
a = torch.tensor([1, 2, 33], dtype = torch.int8)
b = a.float()
```

- 8비트 부호 있는 정수 → 64비트 부동 소수점 수로 변환
```python
c = a.double()
```



