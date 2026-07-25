---
title: annotation 도구
category: general
tags:
  - CV
url: https://velog.io/@kupulau/annotation-도구
created_at: 2024-11-05
related_notes:
---

annotation이란, 이미지 등 시각 데이터에 label을 달아주는 작업이다. 머신 러닝에서 모델을 학습시키기 위해 데이터에 정보를 추가하는 것이다. 예를 들어, object detection 문제를 해결하기 위해서 이미지에 bounding box를 그리고 그 class가 무엇인지 입력하는 것이다. 
이러한 annotation 작업을 수월하게 할 수 있는 도구들을 살펴보겠다. 

<br>

### LabelMe
- MIT CSAIL(Computer Science Artificial Intelligence Laboratory)에서 공개한 image data annotation 도구를 참고해 만든 오픈소스 소프트웨어
- 이미지를 열어서 bounding box, label을 작성하고 결과를 json 파일로 저장할 수 있다.
- python으로 작성되어 있어서 필요한 기능이 있다면 추가할 수 있다.
- 다수의 사용자가 사용할 수 없기 때문에 공동작업이 어렵다.
- object, image에 대한 속성을 부여할 수 없다.

<br>

### CVAT
- Intel OpenVINO 팀에서 제작한 공개 computer vision 데이터 제작 도구
- `Create a new task` → Label 등록 → 이미지 업로드 → bbox/라벨링 작업 → `Actions` > `Export task dataset`


- 단축키
`n` : bbox 새로 생성
`ctrl+s` : 저장
`delete` : bbox 삭제 (맥의 경우 `fn`키와 함께)
이외에도 필요한 단축키를 `Settings` > `Shortcuts`에서 확인하거나 등록할 수 있다.


- 웹 환경에서 사용 가능 (Chrome이 가장 좋음) [링크](https://www.cvat.ai/)
- assignee, reviewer를 지정하여 공동 작업을 진행할 수 있다.
- automatic annotation 기능이 잇다.
- model inference가 느리다.
- object, image에 대한 속성을 부여할 수 없다.

<br>

### Hasty Labeling Tool
- (free credit 소진 이후) 유료
- semi-automated annotation 기능 제공
- assignee, reviewer를 지정하여 multi-user로 사용할 수 있다.
- 도구 커스터마이징 불가

<br>

### LabelImg
- [링크](https://github.com/HumanSignal/labelImg) 
- 이미지와 annotation을 Pascal VOC format의 xml 파일을 한 폴더에 넣어두고 실행하면 이미지 위에 표시되는 bounding box와 라벨을 보면서 작업할 수 있다.  