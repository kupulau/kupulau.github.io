---
title: inductive bias와 representation
category: general
tags:
  - AI
  - ML
  - DL
url: https://velog.io/@kupulau/inductive-bias와-representation
created_at: 2024-09-14
related_notes:
---

**inductive bias(귀납적 편향)** 은 머신 러닝에서 모델이 학습을 할 때 데이터에서 특정 패턴을 찾으려는 경향 혹은 특정 방식으로 데이터를 해석하려는 가정이다. 예를 들어, 선형 회귀 모델은 데이터에서 직선 형태의 관계를 찾으려고 하고, decision tree 모델은 데이터를 decision tree 구조로 분류하고자 한다. 

쉽게 말해 모델은 데이터에 대한 자신만의 **편견** / **선입견** 을 가지고 있다는 것으로 이해할 수 있다. 

inductive bias라는 개념이 없다면, 모델은 데이터에서 패턴을 학습하는 대신 데이터 자체를 그냥 암기하게 된다. 즉, 새로운 데이터에 대한 예측 능력이 크게 떨어진다. (→ overfitting) 또한 우리가 inductive bias를 이해해야 하는 이유는, 모델이 각기 다른 inductive bias를 가지고 있기 때문에, 이에 따라 어떤 문제에 어떤 모델을 적용해야 할지를 결정할 수 있기 때문이다. 


**representation(표현)** 은 모델이 데이터를 다루는 방식 혹은 데이터가 모델 내부에서 변환되는 방식이다. 모델이 이해하는 방식으로 데이터를 변환하는 것이다. 

예를 들어, 이미지는 0~255 사이의 값을 가지는 픽셀들로 이루어져 있는데 이는 모델이 바로 의미를 파악하기 힘든 형태이다. 이 때 Convolutional Neural Network(CNN)는 이미지를 처리하고 패턴을 찾기 위해 feature map을 사용한다. feature map을 통해 모델은 이미지의 중요한 특징을 자동으로 학습한다. 

텍스트의 경우, 인간은 텍스트를 보고 바로 이해할 수 있지만 모델은 단어 자체를 이해하지 못한다. 그래서 모델은 텍스트를 벡터(숫자)로 변환하고 연산하여 단어들 사이의 관계를 이해한다.
 
주식 차트 등의 시계열 데이터 또한 모델이 숫자 나열로 입력받을 때는 이해하기 힘들지만, RNN이나 LSTM과 같은 구조가 시간의 순서를 잘 반영하는 representation을 만들면 시간에 흐름에 따른 데이터의 패턴을 잘 포착할 수 있다.

머신 러닝에서는 사람이 직접 데이터를 어떻게 표현할지 feature engineering을 하는 것이 중요했지만, 딥러닝에서는 데이터를 학습하는 과정에서 자동으로 representation을 찾아낸다.

⭐️ **한마디로 요약하면, representation은 inductive bias의 산물이다.** 