---
title: correlation coefficients
category: statistics
tags:
  - statistics
url: https://velog.io/@kupulau/correlation-coefficients
created_at: 2024-08-08
related_notes:
---

상관계수 (correlation coefficients) 는 변수 간의 상관 관계의 정도를 수치적으로 표현한 통계값이다. 상관계수의 종류에는 Pearson, Spearman, Kendall 등 여러 가지가 있다.

### Pearson correlation coefficient
> $r_{xt}$ = $\frac{\sum_{i=1}^{n} (x_i - \bar{x})(t_i - \bar{t})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2\sum_{i=1}^{n} (t_i - \bar{t})^2}}$

- x와 t가 함께 변하는 정도를 x가 변하는 정도와 t가 변하는 정도로 나누어 두 변수의 관계를 파악
- `np.corrcoef(x, t)` 를 이용하여 상관계수를 구할 수 있다.
```python
import numpy as np

# x = a data array of one variable (ex) 공부 시간)
# t = a data array of the other variable (ex) 성적)
correlation_matrix = np.corrcoef(x, t)

print(correlation_matrix)   
# [[1.        0.8979]
#   0.8979    1.   ]]
```
- 위의 코드 예시처럼 x와 t가 1d array라면 np.corrcoef의 결과가 2 x 2 matrix로 리턴되는데 1행 1열, 2행 2열은 자기 자신 ($x$ vs $x$, $t$ vs $t$) 과의 상관 계수이므로 당연히 1이 나오고, 1행 2열과 2행 1열의 0.8979가 x와 t 사이의 Pearson correlation coefficient 이다.

<br>

### 상관계수의 해석
상관계수는 보통 -1 ~ 1의 값으로 나타나는데 절대값이 0에 가까울수록 변수 간의 상관 관계가 없음을 나타내고, -1에 가까울수록 음의 상관 관계, 1에 가까울수록 양의 상관 관계가 있음을 의미한다.

<br>

### pandas DataFrame과 상관 계수
판다스 데이터 프레임은 상관 계수를 계산해주는 method를 제공한다.
```python
df.corr(method='pearson', numeric_only=True)
```
`method`에 'pearson', 'kendall', 'spearman'을 지정하여 피어슨 상관계수, 켄달 상관계수, 스피어만 상관계수를 계산할 수 있으며 default는 'pearson'이다. 
`numeric_only=True`를 설정하면 테이블 내의 숫자 데이터만 고려하여 상관 계수를 계산해준다. 
