---
title: Optical Character Recognition
category: deep learning
tags:
  - CV
url: https://velog.io/@kupulau/Optical-Character-Recognition
created_at: 2024-10-29
related_notes:
---

OCR (**O**ptical **C**haracter **R**ecognition) 이란, 이미지 내의 글자를 인식해서 읽어내는 태스크이다. 글자가 적혀 있는 텍스트 박스의 위치를 찾고, 텍스트 박스 내에 전사된 글자가 무엇인지를 판별할 수 있어야 한다. STR (**S**cene **T**ext **R**ecognition) 은 OCR의 실생활 적용 예라고 할 수 있는데, 단순히 종이 등에 적힌 글을 판별하는 것에서 더 발전하여 길거리의 간판 등에서도 글자를 읽어내는 것을 의미한다. 

object detection이 객체의 위치를 검출한 후 그 클래스를 예측하는 태스크라면, OCR은 텍스트의 위치를 검출한 후 text recognition 작업이 이어진다. OCR만의 특이점은, 1) 밀도가 높음, 2) 극단적 종횡비, 3) 특이 모양 (구겨짐, 휘어짐, 세로쓰기 등) 등이 있다.   

<br>

### 글자 영역 표현법
- RECT
(x1, y1, height, width) or (x1, y1, x2, y2)


- RBOX
(x1, y1, height, width, $\theta$) or (x1, y1, x2, y2, $\theta$)


- QUAD : bounding box의 좌상단부터 시계 방향으로 4개의 좌표
(x1, y1, x2, y2, x3, y3, x4, y4)


- Polygon
(x1, y1, ..., xN, yN)

<br>

### Modules
OCR은 크게 text detector, text recognizer, serializer, text parser 4가지의 모듈로 이루어져 있다.

#### Text detector
텍스트 영역의 위치 정보를 파악하는 모듈이다. 

#### Text recognizer
텍스트 영역 내의 글자를 인식하고 전사하는 모듈이다. input은 이미지지만 output은 텍스트인, computer vision과 natural language processing의 교집합의 영역이다. 

#### Serializer
OCR 결과값을 자연어 처리하기에 편하게 일렬로 정렬하는 모듈이다. 단락끼리 묶고, 단락 간에 정렬하고, 단락 내에서는 좌상단에서 우하단으로 정렬한다. 정렬한 값에 금칙어 처리, 요약, 글자 의미 파악 등의 자연어 처리 모듈을 연결하여 활용할 수 있다. 

#### Text parser
기존에 정의된 key들에 대해 value를 연결한다. 예를 들어 명함에 대한 OCR을 해서 `홍길동`, `010-1234-5678`, `abc@naver.com` 과 같은 정보를 추출했다면, 이들을 각각 `이름`, `전화번호`, `이메일` 로 연결짓는 작업이다.
이 과정에서 BIO tagging이라는 작업을 하는데, 단어의 처음 (**B**eginning), 중간 (**I**nside), 마지막 (**O**utside) 를 태깅하여 단어별로 구분짓는다. 

<br>

### TrOCR
TrOCR (**Tr**ansformer-based **O**ptical **C**haracter **R**ecognition with pre-trained models) 은 이름 그대로 OCR 태스크에 transformer를 접목시킨 것이다. 
사전 학습된 trasformer 모델을 encoder로, 사전 학습된 language model을 decoder로 사용한다. 전처리/후처리를 하지 않고도 좋은 성능을 낸다. 

<br>

### DTrOCR
DTrOCR (**D**ecoder-only **Tr**ansformer-based **O**ptical **C**haracter **R**ecognition) 는 이미지 encoder 없이 이미지 embedding을 더 큰 text decoder에 넣은 모델ㄹ이다. 파라미터 개수가 기존 TrOCR보다 적고, 더 큰 사전 학습 데이터셋을 사용하여 성능을 높였다. 

<br>

### MATRN
기존의 OCR 모델은 글자가 가려지거나 잘려서 온전한 형태를 유지하지 못한 이미지 혹은 해상도가 낮은 이미지에서는 정확도가 낮다. 그러나 multi-modal적인 접근을 하면 그러한 한계에서도 가려지거나 잘려나간 부분을 추측하여 보다 정확도가 높게 글자를 인식할 수 있다.  
이러한 접근법으로 글자를 인식하고자 한 것이 MATRN (**M**ulti-mod**A**l **T**ext **R**ecognition **N**etwork) 모델이다. MATRN은 feature extractor, multi-modal feature enhancement, output fusion의 3가지 부분으로 구성되어 있다.


