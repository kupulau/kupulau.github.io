---
title: MMDetection 사용법
category: deep learning
tags:
  - CV
url: https://velog.io/@kupulau/MMDetection-사용법
created_at: 2024-10-03
related_notes:
---

MMDetection은 PyTorch를 기반으로 object detection 등에 널리 쓰이는 딥러닝 라이브러리이다. 
object detection은 classification 뿐만 아니라 물체가 이미지 내에서 어느 위치에 있는지까지 알아내야 하는 작업이다보니 직접 구현하려면 굉장히 복잡하고 어렵다. 이 때 MMDetection이나 Detectron2를 이용하면 configuration 설정을 하는 것만으로도 object detection을 수행할 수 있다. 

### Import
```python
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apls import train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import get_device
```

<br>

### Configuration 파일 다루기

```python
# configuration file 들고오기
# 틀이 갖추어진 configuration file을 상속받아 필요한 부분만 수정해서 사용한다.
cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

route = './dataset/'     # 이미지 (annotation) 파일이 있는 경로

# configuration 수정하기
classes = ("General Trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
cfg.model.roi_head.bbox_head.num_classes = len(classes)

# training set 설정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = route
cfg.data.train.ann_file = route + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512, 512)   # resize할 크기를 입력

# validation set 설정
cfg.data.val.classes = classes
cfg.data.val.img_prefix = route
cfg.data.val.ann_file = route + 'val.json'
cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

# test set 설정
cfg.data.test.classes = classes
cfg.data.test.img_prefix = route
cfg.data.test.ann_file = route + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

# 학습 설정
cfg.data.samples_per_gpu = 4
cfg.seed = 2020
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
cfg.optimizer_config.grad_clip = dict(max_norm = 35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()
```

<br>

### dataset 정의
```python
datasets = [build_dataset(cfg.data.train)]
```

<br>

### model 정의
```python
model = build_detector(cfg.model)
model.init_weights()    # 가중치 초기화
```

<br>

### 학습
```python
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
```

<br>

### Custom backbone 모델 등록하기
MMDetection에서 다양한 모델을 이용할 수 있는데, 원하는 모델이 없고 직접 구성해서 사용하고 싶다면 custom backbone 모델을 등록해서 사용할 수 있다.

```python
import torch.nn as nn
from ..builder import BACKBONES

@BACKBONES.register_module()
class MyModel(nn.Module):

	def __init__(self, args):
    	pass
        
    def forward(self, x):    # tuple을 return해야 함
    	pass
        
from .mymodel import MyModel    # mmdetection/mmdet/models/backbones/mymodel.py로 저장된 것을 불러옴
```
모델을 등록, import한 후 아래와 같이 configuration을 변경해 backbone model을 사용하면 된다.
```python
model = dict(
	...
    backbone = dict(
    type='MyModel',
    args = 'arg1')
)
```

<br>

### References
https://github.com/open-mmlab/mmdetection
https://mmdetection.readthedocs.io/en/v2.22.0/_modules/mmdet/apis/train.html