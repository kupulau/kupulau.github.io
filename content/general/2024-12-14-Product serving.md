---
title: Product serving
category: general
tags:
  - dev
url: https://velog.io/@kupulau/Product-serving
created_at: 2024-12-14
related_notes:
---

프로그램이나 모델을 개발하면, 그것을 클라이언트가 사용할 수 있도록 제품으로 배포해야 할 것이다. 이를 serving이라고 한다. ChatGPT에게 prompt로 요청이나 질문을 해서 답을 받는 것을 간단한 예시로 들 수 있겠다.

Serving은 크게 2가지, batch serving과 online (real time) serving 방식을 많이 사용한다. 어떤 상황에서 어떤 serving을 해야하는지 정답이 있지는 않고, 문제 상황, 제약 조건, 인력, 데이터 저장 형태, 레거시 유무 등에 따라 결정할 수 있다. 

### Batch serving
Batch serving은 데이터를 일정 묶음 단위로 서빙하는 것을 의미한다. 
실시간 응답이 중요하지 않은 경우, 대량의 데이터를 처리할 때, 정기적인 일정으로 수행할 때, 인력이 적을 때, RDB, 데이터 웨어하우스를 이용할 때, 유용하다.
ex) Netflix 콘텐츠 추천 (n시간 단위로 예측 후에 DB에 저장한다. 그리고 DB에 있는 예측 결과를 불러와 서빙)

#### Batch pattern
![](https://velog.velcdn.com/images/kupulau/post/e2809d7d-8eb2-476d-a79f-b29ef33bd465/image.png)
주기적으로 데이터를 모아 예측하고 그 결과를 DB에 저장하고 서비스 서버에서는 DB에서 결과를 읽어와 서빙하는 패턴이다. 크게 3개의 파트로 나눠서 생각할 수 있다.

- job management server
작업을 총괄하는 서버
Apache airflow 등을 활용
특정 시간에 주기적으로 batch job을 실행시키는 주체


- job
어떤 작업 실행에 필요한 모든 활동
python script, docker image 등을 활용
ex) model load, data load


- data
DB, 데이터 웨어하우스

Batch 패턴은 예측 결과를 실시간으로 얻을 필요가 없는 경우, 대량의 데이터에 대한 예측을 하는 경우, 예측 실행이 시간대별, 월별, 일별로 스케줄링해도 괜찮은 경우에 유용하게 사용할 수 있다. 

Batch 패턴을 활용하면 기존에 사용하던 코드를 재사용할 수 있고, API 서버를 굳이 개발하지 않아도 된다. 서버 리소스를 유연하게 관리할 수 있지만, 별도의 scheduler(ex) Apache airflow)가 필요하다.

<br>

### Online (real time) serving
Online (real time) serving은 클라이언트가 요청할 때 바로 만들어서 서빙하는 것을 의미한다. 
즉각적으로 응답을 제시해야 하는 경우, 개별 요청에 대한 맞춤 처리가 중요할 때, 동적인 데이터에 대응할 때 유용하다.
다만 API 서버, 실시간 처리 등의 기술이 필요하다.
데이터는 서비스 요청 시 같이 제공하게 된다. 
ex) 유튜브 추천 시스템 (새로고침 할 때 마다 바뀜), 번역 등

#### Web single 패턴
![](https://velog.velcdn.com/images/kupulau/post/f8f32f15-d972-4dc2-888c-7f1ee2bbfe85/image.png)

API 서버 코드에 모델을 포함시킨 뒤에 배포해서 예측이 필요한 곳(클라이언트, 서버 등)에서 직접 서비스를 요청하여 받아오는 패턴이다. 크게 4개의 파트로 나눠서 생각할 수 있다.

- Inference server
FastAPI, Flask 등으로 단일 REST API 서버를 개발 후 배포한 것
API 서버가 실행될 떄 모델을 로드하고, 로직 내에 전처리도 같이 포함


- Client
서비스를 요청하는 창구


- Data
요청할 때 데이터를 같이 담아 요청


- Load balancer
트래픽을 분산시켜 서버에 과부하가 걸리지 않도록 해줌
Ngix, Amazon ELB 등을 사용

Web single 패턴은 inference server를 빠르게 출시하고 싶은 경우, 예측 결과를 실시간으로 얻어야 하는 경우에 사용한다. 보통은 이를 기본으로 삼고 다른 패턴들을 적용하는 식으로 확장해 나간다.

Web single 패턴은 보통 하나의 프로그래밍 언어로 진행할 수 있고, 아키텍처가 단순하기 때문에 처음 사용할 때 좋은 방식이다. 그러나 구성 요소가 하나라도 바뀌면 전체 업데이트가 필요하고, 모델이 큰 경우 로드에 시간이 오래 걸릴 수 있다는 단점이 있다. 

#### Synchronous 패턴
synchronous 패턴은 하나의 작업이 끝날 때까지 다른 작업을 시작하지 않고 기다린 다음, 작업이 끝나면 새로운 작업을 시작하는 방식이다. 대부분의 REST API는 synchronous 패턴을 사용하여 서빙한다. 예측의 결과에 따라 클라이언트의 로직이 즉각적으로 달라져야 하는 경우에 사용하는 패턴이다. architecture 및 workflow가 단순하다는 장점이 있다. 그러나 동시에 여러 요청이 올 경우 처리하기 어렵다는 단점이 있다.

#### Asynchronous 패턴
asynchronous 패턴은 하나의 작업을 시작하고 결과를 기다리는 동안 다른 작업을 할 수 있는 패턴을 말한다. 클라이언트와 예측 서버 사이에 메시지 시스템인 queue를 추가하여 메시지를 저장(push)하고 다시 가져와서 (pull) 작업을 수행하는 방식으로 여러 요청이 들어와도 병렬적으로 처리할 수 있게 된다. 이런 작업을 도와주는 tool의 예로는 Apache Kafka 등이 있다. 
이 패턴에서는 클라이언트와 예측 프로세스가 분리되어 클라이언트가 예측을 기다릴 필요가 없으나 queue 시스템을 따로 만들어야 해서 전체적으로 구조가 복잡해지고, 완전한 실시간 예측에는 적절하지 않다는 점을 고려해야 한다.

<br>

### References
https://mercari.github.io/ml-system-design-pattern/