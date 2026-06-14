---
title: Backpropagation
category: deep learning
tags:
  - DL
url: https://velog.io/@kupulau/Backpropagation
created_at: 2024-08-13
related_notes:
---

역전파 (屰전파; backpropagation) 는 딥러닝에서 신경망을 학습시키는데 사용되는 테크닉이다.

모델 학습은 크게 2가지 phase로 구성되는데, 1) 주어진 input, weight, bias를 그대로 계산하여 output을 예측하는 과정인 **forward pass**, 2) 그 후 예측과 정답 값 사이의 loss를 계산하여 그를 바탕으로 원하는 값이 출력되도록 weight 등을 조정하는 **backpropagation**이다. 

### 역전파 진행 과정

역전파에서는 weight를 조정하기 위해서 loss에 대해 각 단계의 변수로 미분을 한다. 아래 그림은 역전파의 과정을 잘 보여준다.

![](https://velog.velcdn.com/images/kupulau/post/c1a52b1a-f981-406d-8ee4-7bf3f37dd94c/image.png)

- upstream gradient : predicted z와 groundtruth z 사이의 차이인 Loss $L$ 을 output variable z로 미분한 것
- local gradient : output z를 input x, y로 미분한 것
- downstream gradient : upstream gradient와 local gradient의 곱. 결국 loss를 각 input parameter x, y로 미분한 것 

<br>

### 역전파 연산의 패턴

- add gate (→ gradient distributor)
더하기 연산자를 지나면 upstream gradient 값이 그대로 downstream gradient로 흘러간다.
- mul gate (→ swap multiplier) 
곱하기 연산자를 지나면 downstream gradient 값이 upstream $\times$ 상대방의 변수 값이 된다.
- copy gate (→ gradient adder)
copy 연산자를 지나면 downstram gradient 값이 upstream gradient들의 합이 된다.
- max gate (→ gradient router)
maximum 연산자를 지나면 operation이 일어나는 방향으로만 (여러 input 값들 중 가장 큰 input값이 있는 방향으로만) upstream gradient에서 downstream gradient로 흘러간다.

<br>

### References
https://hul980.tistory.com/29