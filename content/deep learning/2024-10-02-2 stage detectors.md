---
title: 2 stage detectors
category: deep learning
tags:
  - CV
url: https://velog.io/@kupulau/2-stage-detectors
created_at: 2024-10-02
related_notes:
  - "[[2024-10-07-Advanced 2 stage detectors]]"
---

object detection task를 하기 위한 딥러닝 모델들의 발전 과정을 알아보자.

2 stage detector는 인간이 사물을 인지할 때와 비슷한 방식으로 동작하는 detector이다. 사람이 물체를 보고 1) 위치를 파악한 뒤 2) 물체의 종류를 판별 하는 것처럼 이 두 단계를 거쳐 object detection을 하는 것이다. 

이러한 2 stage detector는 R-CNN → SPPNet → Fast R-CNN → Faster R-CNN 과 같이 발전해왔다.



### Region-based Convolutional Neural Network (R-CNN)
[R-CNN](https://arxiv.org/abs/1311.2524)이 object detection을 하는 과정은 다음과 같다.

1. selective search (처음에는 이미지에서 잘게 segmentation을 한 뒤 연관이 있는 픽셀들을 병합하면서 region을 찾고 bounding box를 설정하는 방법) 를 이용해 2000개의 Region of Interest (RoI) 를 추출한다.
\* 기존에는 창문을 여닫듯이 미리 설정한 다양한 크기와 형태의 bounding box를 이동시키며 region을 찾는 sliding window라는 방법을 사용했으나, 비효율적이어서 더 이상 사용하지 않게 됨
2. RoI를 모두 동일한 크기로 조절 (CNN의 마지막인 FC layer의 입력 사이즈가 고정이기 때문)
3. RoI를 CNN에 넣어 feature 추출 (Pretrained AlexNet 활용)
\* IoU > 0.5인 positive sample 32개와 IoU < 0.5인 negative sample 96개로 구성된 dataset
4. 추출한 feature를 SVM에 넣어서 classification을 한다. → 클래스(+배경 여부)와 confidence score가 output으로 나온다. 
\* GroundTruth 32개를 positive sample, IoU < 0.3인 negative sample 96개로 구성된 dataset
\* hard negative mining : hard negative sample은 객체가 없음에도 불구하고 객체와 비슷하게 생겨서 모델이 객체로 잘못 예측하는 배경을 의미한다. 이러한 샘플들을 강제로 다음 batch의 negative sample로 활용함으로써 False Positive를 줄이는 기법이다.
5. Regression을 통해 CNN에서 나온 feature로 bounding box 예측 (Bbox regressor)
\* IoU > 0.6인 positive sample들을 대상으로 bbox를 예측했고 이 때 MSE Loss function을 사용
 
Q. negative sample을 사용하는 이유?
A. negative sample은 이미지에서 객체가 포함되어 있지 않은 부분을 의미하는데, 이를 사용해서 모델이 다양한 배경 상황에서 객체가 없는 경우를 잘 인식하고 구별할 수 있도록 학습시키기 위해서이다. negative sample을 이용하면 False Positive를 줄일 수 있다. 

Q. negative sample이 positive sample의 3배수인 이유?
A. 실제 이미지 데이터셋에서는 객체가 포함되지 않은 (배경) 영역이 더 많기 때문에, 이러한 데이터의 불균형을 반영해서 negative sample을 더 많이 포함시킴으로써 모델이 다양한 배경을 제대로 학습해서 False Positive를 줄이고 모델의 일반화 성능을 향상시킨다. "3"배수인 이유는 이 정도 비율이 실험적으로 성능 향상에 유리한 균형을 맞추는 것으로 확인되었기 때문이라고 한다.

이 연구에서는 2000개의 RoI가 각각 CNN을 통과하기 때문에 연산이 오래 걸리고, 강제로 RoI를 동일한 크기로 조절함으로써 성능이 하락할 가능성이 있다. 또한 CNN, SVM classifier, bounding box regressor가 모두 따로 학습되기 때문에 비효율적이고, end-to-end가 갖춰지지 못했다는 것이 아쉬운 점이다.

<br>

### Spatial Pyramid Pooling Network (SPPNet)

[SPPNet](https://arxiv.org/abs/1406.4729)은 입력 이미지를 고정된 크기로 강제로 조정해야하고 2000번의 CNN을 통과해야 하는 R-CNN의 단점을 보완한 모델이다. 

SPPNet에서는 Spatial Pyramid Pooling 방식을 도입해서 입력 이미지의 크기가 다양하더라도 CNN의 마지막에서 고정된 길이의 feature vector를 생성할 수 있도록 했다. 기존의 max pooling과 비슷하지만, 고정된 크기의 pooling window를 사용하는 대신 다양한 크기의 pooling window를 적용한다. 

1. 입력 이미지가 CNN을 통과하여 feature map을 출력한다. feature map의 크기는 입력 이미지의 크기에 따라 다르다.
2. 다양한 크기의 pooling layer (1 $\times$ 1, 2 $\times$ 2, 4 $\times$ 4 등) 를 적용하고 그 결과들을 합쳐서 고정된 길이의 feature vecot로 결합한다.
3. feature vector를 Fully-Connected layer로 전달해서 detection, classification 등의 작업을 수행한다. 

SPPNet은 다양한 크기의 이미지를 CNN에 직접 입력할 수 있으므로 강제로 리사이징을 해서 성능이 떨어질 우려가 없다. 게다가 CNN에서 나온 feature map을 한 번만 계산하면 되어서 계산 비용이 크게 줄어들었다.

하지만 여전히 CNN, SVM, bounding box regression을 모두 따로 학습해야 한다는 문제점이 남아 있다. 

<br>

### Fast R-CNN

[Fast R-CNN](https://arxiv.org/abs/1504.08083)은 CNN, SVM, regressor를 따로 학습해야 하는 기존의 모델을 보완하기 위해 고안되었다.

1. VGG16을 사용하여 이미지를 입력하고 feature map을 추출한다. 
2. RoI projection
먼저 selective search를 통해서 객체가 있을 것으로 예상되는 후보 영역인 region proposal을 생성한다. 그리고 1에서 추출한 feature map에 대해 각 region proposal의 위치를 기반으로 해당 영역을 매핑한다. 
3. RoI pooling 
각 candidate region 여러 개의 부분 영역으로 나누 후, 각 부분에서 가장 큰 값을 추출하여 (→ max pooling) 고정된 크기의 feature vector로 변환한다.
4. Fully-Connected layer
3에서 얻은 feature vector가 FC layer를 통과한 후 softmax classifier로 각 RoI에 대한 클래스 점수가 계산되고 bounding box regression을 수행한다. (클래스 개수 C개 + 배경 1개)

Dataset은 IoU > 0.5인 positive sample을 25%, 0.1 < IoU < 0.5 인 negative sample을 75% 로 구성했다. 
사용하는 loss function의 경우 classificaion task에서는 cross entropy, Bbox regressor에서는 Smooth L1이라는 손실 함수를 사용했다. 최종 손실은 두 손실을 합산하여 계산되었다. 

따라서 Fast R-CNN은 CNN, SVM, regressor의 모든 구성 요소가 함께 학습 및 최적화되는 장점이 있다.

<br>

### Faster R-CNN

[Faster R-CNN](https://arxiv.org/abs/1506.01497)은 **R**egion **P**roposal **N**etwork (RPN) 을 도입해서 Fast R-CNN보다 성능을 향상시키고 object detection의 모든 과정을 end-to-end로 다룬 모델이다. 아래 과정에서 2~5가 RPN의 동작이다. 

1. 이미지를 CNN에 입력하여 feature map을 추출한다.
2. 생성된 feature map 위에서 sliding window 방식으로 feature map을 스캔하며 후보 영역을 생성한다.
3. 각 sliding window 위치에서 다양한 크기와 비율을 가진 (9개의) anchor box를 생성한다.
4. 각 anchor box에 대해 객체가 존재할 가능성을 나타내는 objectness score를 계산한다.
5. bounding box regression을 사용해 anchor box가 실제 객체의 경계와 얼마나 차이가 나는지 보정한다.
6. 이렇게 RPN으로 만들어진 후보 영역은 Fast R-CNN 방식으로 처리된다. (RoI pooling → classification & bounding box regression)

★ **N**on-**M**aximum **S**uppression (NMS)
NMS는 다수의 겹치는 bounding box가 생성될 때 가장 적합한 bounding box를 선택하고 나머지를 제거하는 과정이다. 먼저, 모든 bounding box를 객체일 확률(score)에 따라 높은 순서대로 정렬해서 가장 높은 score를 가진 bounding box를 선택한다. 그리고 다른 bbox들과 선택된 bbox와의 IoU 값을 계산해서 일정 기준 이상인 다른 bbox들은 제거한다. 남은 bbox 중에서 가장 높은 score를 가진 박스를 다시 선택하고, 이 과정을 모든 box에 대해 NMS가 적용될 때까지 반복한다. Faster R-CNN에서는 각 sliding window 마다 9개의 anchor box를 생성하므로, RPN 이후와 최종 detection 단계에서 이 NMS 과정이 적용된다. 이 논문에서는 IoU가 0.7 이상인 proposal 영역들을 중복된 영역으로 판단해서 제거하였다.

Faster R-CNN은 RPN을 도입함으로써 CNN 내부에서 region proposal을 생성하고 classification 을 동시에 처리할 수 있게 되었다. 그래서 이전의 모델들보다 훨씬 빨라졌고, 전체 모델이 더 잘 최적화되고 성능이 향상되었다. 

<br>

### Framework

![](https://velog.velcdn.com/images/kupulau/post/984cb429-948f-4fa3-b60f-b1dbf2db730b/image.png)
[Image credit](https://arxiv.org/abs/1906.07155)

2 stage detctor 모델은 크게 backbone, neck, head로 구성된다.

- Backbone : object detection 모델의 기초가 되는 부분으로, 이미지에서 특징을 추출한다. 일반적으로 pretrained CNN (ex) ResNet-50) 등이 backbone으로 사용된다.

- Neck : backbone에서 추출된 feature map을 재구성하여 head와 연결한다. 다양한 크기와 비율을 가진 물체를 잘 탐지하기 위해서, 멀티 스케일의 특징을 잘 처리하도록 가공한다.

- Head : neck에서 가공된 feature map을 입력받아서 각 객체의 위치와 클래스를 분류한다. 
  - DenseHead : feature map의 dense location을 수행 
  - RoIHead : RoI feature를 입력받아 classification, box regression

<br>

더 발전된 2 stage detector에 대한 것은 이 [[2024-10-07-Advanced 2 stage detectors|포스팅]]을 참고

