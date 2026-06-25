---
title: Multimodal
category: general
tags:
  - AI
url: https://velog.io/@kupulau/Multimodal
created_at: 2024-09-02
related_notes:
---

Multimodal learning이란?
여러 가지 형태의 데이터 (audio, vision, text 데이터 등)를 모두 함께 이용하여 학습을 진행하는 것이다. 

multimodal learning에 앞서 여러 가지 형태의 데이터에 대해 알아보자.

- 2D image
  - an intensity value of each pixel in 2D array structure 
  - 3D tensor (Height $\times$ Width $\times$ Channel (color))
- video : a stack of image frames
- 3D image : multi-view images, voxel, part assembly, point cloud, mesh, implicit shape
- text → dense vector를 assign한 token
- sound : waveform, power spectrum, spectrogram


### CLIP
multi-modal alignment를 잘 구현한 모델 중 하나는 CLIP (**C**ontrastive **L**anguage-**I**mage **P**re-training) 이다. 
Text encoder와 image encoder를 갖추고 feature embedding을 한 후 text embedding과 image embedding을 비교/대조(→ joint embedding)하여 이미지에서 텍스트를 뽑아내거나 텍스트로 이미지를 검색할 수 있다. 
CLIP에서는 contrastive learning을 이용해 학습을 진행하는데, contrastive learning은 이미지가 주어졌을 때 텍스트와 cosine similarity를 계산하여 유사도가 높은 텍스트는 서로 당겨주고, 매칭되지 않는 것과는 거리를 벌리는 것이다.

pre-trained CLIP의 응용 사례
zeroshot
styleclip
3D avatar

### ImageBIND
ImageBIND는 텍스트, 이미지 뿐만 아니라 다른 sensor data 등과 align하고 관계성을 학습하는 모델이다. 


### Cross-modal translation
하나의 modality에서 다른 modality로 번역을 하는 기법이다. Input과 output의 modality가 변환된다. 
text-to-image generation(ex) DALL-E), sound-to-image synthesis, speech-to-face synthesis 등이 그 예이다.

### LLaVA
이미지가 주어졌을 때 그 이미지에 대한 discussion이 가능한 visual reasoning model
Pre-train된 LLM을 이용
Language model의 사전 지식을 활용하고
학습할 때 projection layer (linear layer)를 배치해서 visual encoder에서 나온 feature가 language model이 이해할 수 있는 token 형태로 변환해준다.

### InstructBLIP
visual reasoning 및 captioning task를 하는 모델
Q-former 모듈을 이용 image embedding을 조건으로 해서 language model이 이해할 수 있는 token 형태로 변환하는 역할
Text instruction이 입력되어서 Q-former에서 image embedding을 token으로 변환할 때 instruction에 연관이 높은 visual feature 형태로 변환해서 다음 단계로 넘긴다. 그 후 fully connected layer를 통과하고 frozen LLM 모델이 대답을 한다.

### X-InstructBLIP
InstructBLIP을 다양한 modality의 encoder로 확장

### Visual programming
Laugnage model을 그대로 활용한 모델. 
이미지가 주어졌을 때 이미지를 어떤 식으로 분해해서 재합성을 해야하는지에 대한 각각의 절차를 계획하고 절차에 대한 프로그램을 생성한다. 
In-context learning을 활용한다. In-context learning은 exmaple을 제공해서 그것을 참고해서 답을 내리도록 학습시키는 것. 

### PaLM-E
이미지가 주어졌을 때 그 이미지를 language token으로 변환하는 부분 뿐만 아니라 text를 generation해서 만든 액션을 컨트롤 파운데이션 모델에 입력해서 로봇을 직접 제어할 수 있는 신호로 변환해준다. 


