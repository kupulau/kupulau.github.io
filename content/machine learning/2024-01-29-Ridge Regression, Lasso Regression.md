---
title: Ridge Regression, Lasso Regression
category: machine learning
tags:
  - ML
  - statistics
url: https://velog.io/@kupulau/Lasso-Regression-Ridge-Regression
created_at: 2024-01-29
related_notes:
---
logistic regression을 scikit-learn을 이용해 구현하기 위해서는, penalty라는 parameter를 정해서 입력해야한다. 공식 문서를 확인해보면 penalty는 None, 'l1', 'l2', 'elasticnet' 중 하나를 선택할 수 있고, default는 'l2'다. 'elasticnet'은 'l1'과 'l2'를 적절히 섞어쓰는 모양인데, 도대체 'l1'과 'l2'가 무엇인가?

우리가 머신 러닝에서 주의해야 할 점들 중 하나는 바로 training set에 모델이 overfit 되지 않도록 하는 것이다. training set에 모델이 overfit되면, test set을 적용할 때 variance가 커지게 된다. 

#### Ridge Regression (L2 regression)
Ridge regression의 주요 아이디어는 모델을 "일부러" training set에도 잘 피팅이 되지 않게 만드는 것이다. 즉, training set에 피팅을 하되 **약간의 bias를 도입하는 것이다.** 이를 통해 bias는 약간 생기지만, variance가 크게 줄어들면서 training set에 overfit되는 것을 막고, test set을 적용했을 때에도 적절한 성능을 내도록 하는 것이다.

bias를 도입할 때는 $\lambda$ $\times$ $slope^{2}$ 항을 더해주면 된다. (여기서 slope는 linear regression의 경우에서 직선의 기울기를 의미) $\lambda$ 가 바로 bias를 도입하면서 생기는 "penalty"를 얼마나 강하게 줄 지 결정하는 값이다. $\lambda$는 0보다 큰 모든 수가 들어갈 수 있지만, 숫자가 커질수록 slope가 0에 근접하게 된다. $\lambda$를 어떤 값으로 채택해야 할지는 바로 알 수 없고, cross validation 등을 통해서 가장 variance가 낮게 나오는 $\lambda$를 선택하면 된다.

#### Lasso Regression (L1 regression)
Lasso regression은 Ridge regression과 거의 비슷하지만, bias를 도입할 때 $\lambda$ $\times$ $slope^{2}$ 대신 $\lambda$ $\times$ |$slope$| 항이 더해진다. 이러한 특성으로 인해 모델이 상대적으로 중요도가 떨어지는 파라미터를 많이 포함하고 있을 때 그러한 파라미터들을 제거할 수 있다. (자세한 과정은 영상 참조)

#### 그렇다면 'l1'과 'l2' 중 어느 것을 선택해야 하는가?
feature들끼리 관련이 적거나 중복되어 불필요한 feature들이 많은 데이터 → 'l1' (Lasso Regression)
feature들끼리 연관성이 큰 데이터 → 'l2' (Ridge Regression)

#### Summary
- sklearn LogisticRegression 모델의 penalty 파라미터는 training set에 overfit되지 않도록 하기 위한 것이다.
- sklearn LogisticRegression 모델의 penalty 옵션인 'l1'과 'l2'는 overfit 문제를 해결하기 위해 bias를 도입한다는 점은 같지만 bias 도입을 위해 추가하는 항이 약간 다르다.
- 유튜브 StatQuest with Josh Starmer 채널이 머신 러닝 공부하는 사람들에게 매우 유용하다!!

<br>

#### References
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://www.youtube.com/watch?v=Q81RR3yKn30
https://www.youtube.com/watch?v=NGf0voTMlcs
https://velog.io/@kupulau/Bias-Variance-tradeoff