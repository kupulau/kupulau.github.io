---
title: image processing
category: deep learning
tags:
  - DL
  - CV
url: https://velog.io/@kupulau/image-processing
created_at: 2024-09-13
related_notes:
---

Computer Vision 분야에서 최적의 모델을 개발하기 위해서는, 모델에 입력되는 데이터 즉 이미지의 전처리가 중요하다. 이미지 전처리는 이미지에서 의미있는 feature와 representation을 추출할 수 있기 때문에 모델 성능과 일반화에 큰 영향을 미치게 된다.


### color space
색을 디지털적으로 표현하고 해석하기 위해 정의된 수학적 모델
ex) RGB, HSV, Lab, YCbCr, Grayscale

- openCV를 이용한 color space 변환

```python
import cv2

# original RGB image
img = cv2.imread('image.jpg')

# RGB -> HSV
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # cv2에서는 RGB를 BGR 순으로 표기

# RGB -> LAB
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# RGB -> YCrCb
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# RGB -> greyscale
img4 = cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY)
```

- histogram smoothing
이미지 픽셀값을 평활화 (equalize) 하는 것이다. 평활화 하기 전 픽셀 값은 특정 구간에 몰려있는데 이를 넓은 구간에 분포하도록 펴주는 것이다. 이를 통해 이미지의 constrast를 개선하고 디테일을 잘 보이게 만들어준다. 
```python
new_img = cv2.equalizeHist(img)
```


- 활용 : skin color detection

<br>

### geometric transform

이미지의 형태, 크기, 위치 등을 변환하는 기법
translation, rotation, scaling, shearing, perspective 등이 있다.

입력 이미지 크기는 출력 feature map의 크기에 영향을 준다.
CNN은 사전 학습 중에 feature map 크기의 패턴을 찾는 방법을 학습한다. 그러나 입력 이미지 크기가 달라지면, 학습된 패턴의 크기도 변경된다. 따라서 이미지 크기가 변경되면 object를 찾지 못하게 될 수 있다. 

- translation (물체의 위치 이동)

>$$\begin{bmatrix} x'\\ y'\\ 1 \end{bmatrix}$$ = $$\begin{bmatrix} 1&0&t_x\\ 0&1&t_y\\ 0&0&1\\ \end{bmatrix}$$ $$\begin{bmatrix} x\\ y\\ 1 \end{bmatrix}$$


```python
import numpy as np

row, col = img.shape[:2]


# x 방향으로 100, y 방향으로 50 shift
matrix = np.float([1, 0, 100], [0, 1, 50], [0, 0, 1]])    
new_img = cv2.warpPerspective(img, matrix, (rcol, row))
```

- rotation

> $$\begin{bmatrix} x'\\ y'\\ 1 \end{bmatrix}$$ = $$\begin{bmatrix} cos \,\theta&-sin \,\theta&0\\ sin \,\theta&cos\,\theta&0\\ 0&0&1\\ \end{bmatrix}$$ $$\begin{bmatrix} x\\ y\\ 1 \end{bmatrix}$$

```python
# (반시계방향으로) 90도 회전
matrix1 = cv2.getRotationMatrix2D((col/2, row/2), 90, 1)
matrix2 = np.vstack([matrix1, [0, 0, 1]])
new_img = cv2.warpPerspective(img, matrix2, (col, row))
```

- resize (scaling)
>$$\begin{bmatrix} x'\\ y'\\ 1 \end{bmatrix}$$ = $$\begin{bmatrix} s_x&0&0\\ 0&s_y&0\\ 0&0&1\\ \end{bmatrix}$$ $$\begin{bmatrix} x\\ y\\ 1 \end{bmatrix}$$

```python
# 가로 세로를 2배로 늘림
matrix = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
new_img = cv2.warpPerspective(img, matrix, (2*col, 2*row))
```

크기가 늘어난 후 새로운 위치는 interpolation을 통해 채워진다.

- perspective transformation (원근 변환)

>$$\begin{bmatrix} x'\\ y'\\ w' \end{bmatrix}$$ = $$\begin{bmatrix} a_{11}&a_{12}&a_{13}\\ a_{21}&a_{22}&a_{23}\\ a_{31}&a_{32}&1\\ \end{bmatrix}$$ $$\begin{bmatrix} x\\ y\\ 1 \end{bmatrix}$$

```python
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
new_img = cv2.warpPerspective(img, matrix, (300, 300))
```

위 코드는 이미지의 특정 지점을 확대하여 보여주는 것을 의미한다. 기존 이미지에서의 네 꼭지점 ((56, 65), (368, 52), (28, 387), (389, 390)) 좌표를 끄트머리로 하여 잘라내고 이 꼭지점을 새로 확대한 이미지의 네 꼭지점 ((0, 0), (300, 0), (0, 300), (300, 300)) 으로 변환한다는 것이다. 따라서 new_img는 img의 특정 부분을 확대한 300 $\times$ 300 이미지가 된다.

<br>

### data augmentation

학습 데이터의 다양성을 증가시켜 모델의 견고성 향상과 과적합을 감소시키는 기법
ex) flip, rotation, crop, color jittering
AutoAugment (데이터셋에 맞춘 최적의 augmentation 정책을 자동으로 탐색), RandAugment (랜덤한 크기로 augmentation의 하위 집합을 무작위로 적용) 등의 advanced augmentation 기법도 있다.


Albumentations를 활용한 data augmentation
```python
import albumentations as A

transform = A.Compose([
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(p=0.2),
    A.RandomCrop(height=224, width=224)])
```

<br>

### normalization

이미지의 픽셀 값을 특정 범위로 스케일링하는 기법이다. 딥러닝 모델의 수렴 속도와 안정성을 개선할 수 있고, 큰 비중을 가진 특징이 학습 과정에서 bias를 만드는 것을 방지할 수 있다. 

- min-max normalization : 이미지 픽셀값을 \[0, 1] 범위로 스케일링

```python
# with PyTorch
from torchvision import transforms

transform = transforms.Compose([
	transforms.ToTensor(),
    transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])
    ])
```

- Z-score normalization (standardization) : 데이터 셋의 평균을 빼고 표준 편차로 나누어 준다.

```python
# with albumentations
from albumentations.pytorch import ToTensorV2
import albumentations as A

transform = A.Compose([
	A.Normalize(
    	mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)), 
    ToTensorV2()
    ])
```

- batch normalization
mini-batch 단위로 입력 데이터를 정규화하는 기법이다. internal covariate shift 문제를 개선할 수 있고, 더 높은 learning rate를 허용한다. initialization에 대한 민감도를 감소시킬 수 있다. 



