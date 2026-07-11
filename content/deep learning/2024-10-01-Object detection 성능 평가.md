---
title: Object detection 성능 평가
category: deep learning
tags:
  - CV
url: https://velog.io/@kupulau/object-detection-overview
created_at: 2024-10-01
related_notes:
  - "[[2024-09-16-모델 성능 평가와 향상]]"
---

Object detection이란, 이미지 내에서 객체가 어디에 있는지, 그리고 그 객체가 무엇인지 판별하는 task이다. 객체가 무엇인지 분류만 하는 classification 문제보다 복잡한 문제라고 볼 수 있다. 

object detection은 자율 주행, OCR, 질병진단, CCTV 등에 활용할 수 있다. 

object detection이 잘 되었는지 평가는 크게 **성능**과 **속도**라는 측면에서 할 수 있다. 
object detection의 성능은 mAP라는 지표로 평가한다. **m**APs는 mean **A**verage **P**recision의 약자이다. mAP에 대해서 이해하기 위해서는 precision과 recall의 개념을 먼저 알고 있어야 한다. 기억이 나지 않는다면 [[2024-09-16-모델 성능 평가와 향상|여기]]서 confusion matrix 부분을 다시 복습하고 오자. mAP의 계산은 TP와 FP를 바탕으로 Precision-Recall curve (PR curve) 를 그리는 것에서 시작한다. 
여러 개의 분류된 항목들이 있을 때, confidence가 높은 순으로 항목들을 나열해서 precision과 recall을 계산하면 아래와 같은 그래프가 나타나는데, 이를 PR curve라고 한다.

![](https://velog.velcdn.com/images/kupulau/post/964b70d3-7f4f-4d31-bf50-1b0e66ba32e8/image.png)

이 PR curve의 아래 면적을 계산한 것이 AP (**A**verage **P**recision) 이고, 이를 여러 클래스들에 대해 평균한 값이 mAP이다. mAP 값이 높으면 우리는 object detection의 성능이 좋다고 평가할 수 있다. 

그렇다면 object detection task에서는 무엇을 기준으로 precision과 recall을 계산하는 것일까? object detection에서는 객체의 위치를 찾아서 bounding box (이하 bbox) 를 설정하는데, 예측한 bbox가 GroundTruth bbox와 일치할수록 잘 detection한 것으로 볼 수 있을 것이다. 이를 확인할 수 있는 지표가 IOU (**I**ntersection **o**ver **U**nion) 이다. 

$IoU = \frac{overlapping \, region}{combined \,  region}$

위와 같은 식으로 계산되는 IoU는 결국 두 bbox의 면적의 총합에서 겹치는 부분의 비율이기 때문에 GroundTruth bbox와 예측한 bbox가 일치할수록 1에 가까운 값이 나올 것이다. 
보통은 IoU 뒤에 두 bbox가 일치하는지 (→ True) 일치하지 않는지 (→ False) 를 결정하는 threshold 값을 붙이는데, 예를 들어 IoU60 이면 IoU 값이 0.6 이상이면 두 bbox가 일치한다고 판단, 0.6 미만이면 일치하지 않는다고 판단한다.
다시 mAP로 돌아와서, IoU threshold 값을 무엇으로 설정했는지에 따라서 mAP50, mAP60, ... 등으로 기준을 설정하여 object detection의 성능을 평가한다.

다음으로 object detection의 속도를 평가할 때는, FPS와 FLOPs라는 두 가지 지표를 활용할 수 있다. 
FPS (**F**rame **P**er **S**peed) 는 1초에 몇 장의 frame을 처리하는지를 나타내는 지표로, FPS 값이 클수록 object detection의 속도가 빠르다고 할 수 있다. FLOPs (Floating Point OPerations) 는 컴퓨터가 1초 동안 수행할 수 있는 부동소수점 연산의 횟수로, FLOPs가 클수록 더 빠르게 처리되는 것이다.


#### References
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html (Image credit)



