---
title: pandas data frame 사용법 모음
category: general
tags:
  - coding
  - python
  - EDA
url: https://velog.io/@kupulau/pandas-data-frame-사용법-모음
created_at: 2024-01-12
related_notes:
  - "[[2024-01-11-EDA]]"
---


```python
import pandas as pd

df = pd.read_csv('df.csv')
```

- 데이터 차원 확인
```python
df.shape
```

- 결측치, 데이터 타입 확인
```python
df.info()
```

- 각 column의 결측치 갯수 확인
```python
df.isnull.sum()
```

- 5 number summary (minimum, Q1, Q2(=median), Q3, maximum)
```python
df.desribe()
```

- 정규표현식에 맞는 문자열 추출 (series)
```python
df['col1'].str.extract(' ([A-Za-z]+)\.', expand=False)   # ' ~~~.' 을 찾겠다. 
# expand = False -> series를 리턴
# expand = True -> data frame을 리턴
```

- 특정 문자열이 포함되었는지 True/False로 나타내기 (series)
```python
df['col1'].str.contains('A', case=False)   # 'A'가 포함되어 있으면 T, 없으면 F
# case = False -> 소문자/대문자 구분 X ('A'도 찾고 'a'도 찾음)
# case = True -> 소문자/대문자 구분함 (즉 'A'만 찾고, 'a'는 찾지 않음) (default)
```

- 값 조작하기 (series.map)
예를 들어  사람들의 인적 사항이 기록된 df가 있고  성별이 'male', 'female'로 표기되어 있는데 이를 남성이면 1, 여성이면 2로 값을 변경하고 싶다고 하자.
```python
gender = {'male' : 1, 'female' : 2}
df['gender'] = df['gender'].map(gender)
```

- 값 조작하기 - 확장 (범주가 많을 때) (.factorize())
바로 위의 예시는 성별의 범주가 'male', 'female'로 2가지밖에 없으니 딕셔너리 하나로 금방 바꿀 수 있었지만, 범주가 많을 경우 딕셔너리로 일일이 지정해주기 힘들 수 있다. 이 때 .factorize()를 사용하면 자동으로 'A' : 0, 'B' : 1, 'C' : 2, ...,  'Z' : 25 이런 식으로 자동으로 변경해준다.
```python
df['col1'].factorize()
```

- 유일한 값 찾기
데이터의 종류가 어떤 것이 있는지 확인할 때 쓰는 것이다.
```python
df.unique()
```

- 데이터 구간 나누기
```python
pd.qcut(df['col1'], q=10)  # 10개의 구간을 생성하고 그 구간에 동일한 데이터 개수가 나누어 들어간다.
pd.cut(df['col1'], bins=3)  # 데이터값을 3등분하여 데이터를 나누어 준다. 
# ex) array1 = [1,2,3,4,5,6,7,8,9] 일 때 pd.cut(array1, 3) 하면 구간이 [1,2,3], [4,5,6], [7,8,9]의 3등분으로 나누어짐
```

<br>

$+$ 정규표현식 관련 보충
정규표현식이란, 특정한 조건의 문자열을 표현하기 위한 식이다. 

|     정규표현식      | 축약 표현 | 의미              |
| :------------: | :---: | --------------- |
|     [0-9]      |  \d   | 숫자 찾기           |
|     [^0-9]     |  \D   | 숫자가 아닌 것 찾기     |
| [ \t\n\r\f\v]  |  \s   | 공백이 있는 문자 찾기    |
| [^ \t\n\r\f\v] |  \S   | 공백이 없는 문자 찾기    |
|  [A-Za-z0-9]   |  \w   | 문자+숫자 찾기        |
|  [^A-Za-z0-9]  |  \W   | 문자+숫자가 아닌 것  찾기 |


<br><br>

#### References
https://wikidocs.net/4308
https://zephyrus1111.tistory.com/409