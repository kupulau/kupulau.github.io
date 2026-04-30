---
title: missingno
category: general
tags:
  - coding
  - python
  - EDA
url: https://velog.io/@kupulau/missingno
created_at: 2024-01-16
related_notes:
---

missingno는 pandas data frame 등과 연동되어 각 데이터에서 결측치가 얼마나 있는지를 한 눈에 볼 수 있도록 시각화하는 기능이 있는 라이브러리이다.

#### 설치
```
pip install missingno
```

#### 사용
```python
import missingno
import pandas as pd

df = pd.read_csv('data.csv')

missingno.matrix(df)

```

#### References
https://chunggaeguri.tistory.com/entry/Visualization-missingno-라이브러리-사용법