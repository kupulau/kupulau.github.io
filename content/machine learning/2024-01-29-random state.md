---
title: random state
category: machine learning
tags:
  - ML
url: https://velog.io/@kupulau/randomstate
created_at: 2024-01-29
related_notes:
---
scikit-learn에서 cross validation을 하기 위해서 k-fold를 이용하면, random_state라는 파라미터가 있다. 이 random_state는 cross validation 할 때 뿐만 아니라 무작위로 (랜덤하게) 데이터를 추출하는 함수에서는 자주 볼 수 있다.

사실 컴퓨터는 랜덤한 값을 만들어낼 수 없다. 보통 컴퓨터가 랜덤한 값을 준다는 것은, 컴퓨터가 현재 시각을 기반으로 랜덤한 값을 계산한다는 것이다. 즉, 특정한 숫자를 기반으로 랜덤한 값을 계산한다는 것이다. 따라서 컴퓨터가 어제 계산한 랜덤값과 오늘 계산한 랜덤값은 다르다. 

이 때 일관적이게 랜덤값을 계산하도록 하는 파라미터가 random_state이다. random_state에는 정수이기만 하면 어떤 값을 입력하든 상관이 없고, random_state를 모두 특정 숫자로 고정하면 어제 계산한 랜덤값과 오늘 계산한 랜덤 값이 일관적이게 할 수 있다.