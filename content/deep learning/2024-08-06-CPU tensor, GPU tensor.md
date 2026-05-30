---
title: CPU tensor, GPU tensor
category: general
tags:
  - pytorch
  - DL
url: https://velog.io/@kupulau/CPU-tensor-GPU-tensor
created_at: 2024-08-06
related_notes:
---

다루는 텐서가 어디에 저장되느냐에 따라 CPU tensor, GPU tensor로 구분되고 CPU에 저장된 것을 GPU로, GPU에 저장된 것을 CPU로 옮기는 것도 가능하다.

<br>

### tensor가 저장된 디바이스 확인
```python
import torch

a = torch.tensor([1, 2, 3, 4])
a.device
```
tensor.device하면 tensor가 CPU에 저장되어 있는지 GPU에 저장되어 있는지 확인할 수 있다.

<br>

### CUDA

CUDA (**C**ompute **U**nified **D**evice **A**rchitecture) 란, GPU에서 수행하는 병렬 처리 알고리즘을 프로그래밍 언어를 사용하여 작성할 수 있도록 하는 기술이다. 

- CUDA 사용 가능 환경 확인
 `torch.cuda.is_available()`
True가 return되면 사용 가능, False는 사용 불가능
 
 - CUDA device 이름 확인하기
 `torch.cuda.get_device_name(device=0)`
 
 - 사용 가능한 GPU 개수 확인
 `torch.cuda.device_count()`

<br>

### CPU, GPU tensor

- CPU tensor
별도로 지정을 하지 않으면 기본적으로 CPU tensor가 생성되는 듯 하다.
```python
a = torch.tensor([1, 2, 3])
```

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

<br>

### AI 분야에서 GPU를 사용하는 이유
- GPU는 수 천 개의 작은 코어를 가지고 있어 대량의 연산을 동시에 수행하는 **병렬 처리 능력**이 뛰어나다.
- GPU의 병렬 처리 능력이 AI 모델의 훈련과 추론 **속도를 향상**시킨다.
- 초기 비용은 크지만, GPU를 사용하면 시간이 단축되고 에너지가 절약되기 떄문에 장기적으로는 **투자 대비 높은 수익**을 얻을 수 있다.

<br>

### Rerefences
https://statisticsplaybook.tistory.com/47
https://ko.wikipedia.org/wiki/CUDA