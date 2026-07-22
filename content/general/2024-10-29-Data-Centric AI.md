---
title: Data-Centric AI
category: general
tags:
  - AI
url: https://velog.io/@kupulau/Data-Centric-AI
created_at: 2024-10-29
related_notes:
---

AI는 크게 모델/알고리즘과 데이터 두 부분으로 구성되어 있다. 보통 AI 교육이나 연구에서는 데이터셋과 평가 방식은 정해진 상태에서 모델을 개선하여 성능을 향상시키도록 한다. 그러나 실제 현업에서는 "영수증 데이터 수집 자동화"처럼 <u>요구사항</u>만 존재하기 때문에, 서비스에 적용되는 AI 개발 업무의 상당 부분이 데이터셋을 준비하는 것이다. 즉, 모델링이 차지하는 비율은 생각보다 적다. (~20%)

![](https://velog.velcdn.com/images/kupulau/post/618909b5-f305-4c18-a375-2ebdbb0f851e/image.png)

따라서 좋은 퀄리티의 데이터를 확보하는 것이 AI 성능 향상에 상당히 중요한 요소이다. 특히나, 모델을 개발해 서비스를 출시한 이후에는 모델 변경하는 것에 대한 비용이 커지기 때문에, 데이터를 정제, 추가하는 것이 성능 개선을 위해 더 중요하다. 이러한 관점으로 AI에 접근하는 것이 Data-centric AI이다.

### DMOps
DMOPs (**D**ata **M**anagement **OP**erations & Recipe**s**) 는 이렇듯 양질의 데이터를 확보하고자 하는 일련의 방법론, 도구, 프레임워크이다.
- Data annotation tool
- Data software tool
- Data labeling tool
- Crowd sourcing : 언제 어디서든 누구나 온라인 플랫폼을 통해 데이터 작업에 참여할 수 있는 방식 

<br>

### 좋은 데이터를 만들기 위해서
AI의 성능 향상을 위해 좋은 데이터를 확보한다는 것의 취지는 좋지만, 사실 정말 어려운 문제이다. 좋은 데이터를 많이 수집하기 어렵고, 라벨링 작업에 대한 명확한 정답이 없고 비용이 크기 때문이다.

따라서 좋은 데이터를 만들기 위해서는 명확한 **라벨링 가이드**를 작성하는 것이 중요하다. 라벨링 가이드를 통해
- 일관성 있게 라벨링된 데이터
- 특이 케이스가 포함된 데이터
- high quality 데이터 
- 골고루 분포되어 있는 (편향되지 않은) 데이터

를 확보할 수 있다. 

<br>

### References
https://www.koraia.org/default/mp5/sub3.php?com_board_basic=read_form&com_board_idx=153 (Image credit)