---
title: make scorer
category: machine learning
tags:
  - ML
  - python
url: https://velog.io/@kupulau/makescorer
created_at: 2024-02-13
related_notes:
---

머신 러닝의 성능을 평가할 때, 어떤 metric을 이용하여 점수를 매길지를 결정해야 한다. 보통은 option 중에 metric을 골라서 점수를 계산할 수 있지만, 때로는 사용자가 필요로 하는 metric을 직접 입력하여 custom scoring이 필요할 수 있다. 이 때 사용하는 것이 scikit-learn의 `make_scorer` 함수이다.

아래의 예시는 RMSLE scoring을 사용자가 정의하여 `make_scorer`로 점수를 매기는 코드이다.

```python
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.model_selection import cross_val_score

# RMSLE로 점수를 매기자.
def RMSLE(y_true, y_model):
    diff = np.log(y_model+1) - np.log(y_true+1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)
    
RMSLE_scorer = make_scorer(RMSLE, greater_is_better=False)
    
scores = cross_val_score(scoring=RMSLE_scorer)    
```

`greater_is_better` 옵션은 말 그대로 계산한 수치가 클수록 점수가 크게 매겨지는 것을 의미하는데, RMSLE의 경우 계산 결과가 작을수록 피팅이 잘 된 것이므로 `great_is_better = False`로 설정해야 한다.


#### References
https://wikidocs.net/84268
https://wooono.tistory.com/204