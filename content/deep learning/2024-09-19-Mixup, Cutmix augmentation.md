---
title: Mixup, Cutmix augmentation
category: deep learning
tags:
  - CV
  - pytorch
url: https://velog.io/@kupulau/Mixup-augmentation
created_at: 2024-09-19
related_notes:
---

### Mixup augmentation

![](https://velog.velcdn.com/images/kupulau/post/74470e49-fee9-4c56-878b-02a967258575/image.png)

mixup augmentation은 학습 데이터에서 두 클래스의 샘플 데이터를 혼합해서 새로운 학습 데이터를 만드는 데이터 증강 기법이다. 위 그림의 예시처럼 고양이와 개의 이미지를 7대 3으로 혼합하는 것이다. 이미지만 혼합하는 것이 아니라, label 또한 혼합한다. 클래스 간의 혼합된 이미지를 학습함으로써, 분류 모델은 더 부드럽고 명확한 결정 경계를 형성하게 된다. 따라서 overfitting을 방지하여 일반화 성능을 더 높일 수 있다. 특히 데이터가 부족하거나 클래스 간의 경계가 모호할 때 유용하게 사용할 수 있는 기법이다. 

코드 상에서 mixup augmentation의 특이점은 다른 전처리들 (crop, flip, normalize 등) 이후에 진행된다는 점이다. 왜냐하면 individual image가 아니라 이미지 데이터들의 batch를 input으로 받아서 처리하기 때문이다.

mixup augmentation을 구현할 수 있는 라이브러리는 여러 가지가 있다. 아래는 `torchvision.transforms.v2`의 `MixUp`을 이용한 코드이다.

```python
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate

num_classes = 100     # 클래스가 100개 있는 데이터

# preprocessing
preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Dataset
dataset = CustomDataset(train_data, train=True, transform=preproc)

# define mixup
mixup = v2.MixUp(num_classes=num_classes)

def collate_fn(batch):
	return mixup(*default_collate(batch))
    
# DataLoader
dataloader = DataLoader(
	train_dataset, 
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn)   
```

아래는 `timm.data`에서 제공하는 `Mixup` 함수를 사용한 예시이다. 이 역시 Dataset에서 transform을 거친 후에 mixup을 적용하면 된다.

```python
from timm.data import Mixup

mixup_fn = Mixup(mixup_alpha=0.4, num_classes=num_classes)
images, targes = mixup_fn(images, targets)
```

<br>

### Cutmix augmentation
![](https://velog.velcdn.com/images/kupulau/post/2da4b7a9-bf28-4577-90f4-d2d9e91df224/image.png)
위는 _Yun et al. (2019)_ 의 그림인데, cutmix가 어떤 것인지, 그리고 mixup이나 다른 기법과 성능 차이가 얼마나 있는지 한 눈에 볼 수 있다. 우선 cutmix란, 그림에서 볼 수 있듯이 서로 다른 클래스의 이미지를 부분적으로 잘라서 합치는 기법이다. mixup이 투명도를 조절해서 겹치는 것이라면, cutmix는 콜라주처럼 잘라서 붙이는 것이다. 여러 task에서 cutmix가 다른 기법들보다 더 좋은 성능을 내고 있는 것을 확인할 수 있다.
`timm.data`를 사용해서 cutmix를 구현할 수 있다. 사실 `Mixup` 함수에서 파라미터를 무엇으로 조정하느냐에 따라 mixup을 구현할 수도 있고 cutmix를 구현할 수도 있다. 

`Mixup` 함수의 파라미터를 (일부) 살펴보면

>`mixup_alpha` : mixup에서의 $\alpha$ 값 설정
`cutmix_alpha` : cutmix에서의 $\alpha$ 값 설정
`cutmix_minmax` : cutmix의 범위 설정
`prob` : mixup 또는 cutmix가 적용될 확률
`switch_prob` : mixup과 cutmix간 전환 확률
`num_classes` : class 수

와 같은 파라미터들을 조정하여 사용할 수 있다. 

예를 들어 mixup만을 사용하려면 `cutmix_alpha = 0`, cutmix만을 사용려면 `mixup_alpha = 0` 으로 설정한다.
```python
# mixup only
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0, prob=0.5, switch_prob=0, num_classes=num_classes)

# cutmix only
mixup_fn = Mixup(mixup_alpha=0, cutmix_alpha=0.2, prob=0.5, switch_prob=0, num_classes=num_classes)
```

<br>


### ⚠️ Caution

mixup/cutmix를 거치면서 레이블이 one-hot 인코딩되기 때문에, correct prediction을 셀 때

```python
images, targets = Mixup(images, targets)
outputs = model(images)

predicted = torch.argmax(outputs, dim=1)
targets_class = torch.argmax(targets, dim=1)
corrected_predictions += (predicted == targets_class).sum().item()
```
와 같이 `.argmax`로 변환한 후 세어야 한다.

<br>

#### References
https://pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html
https://www.kaggle.com/code/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup#Mixup-augmentation (_mixup image credit_)
https://geunuk.tistory.com/456
https://arxiv.org/pdf/1905.04899v1 (_cutmix image credit_)
https://timm.fast.ai/mixup_cutmix