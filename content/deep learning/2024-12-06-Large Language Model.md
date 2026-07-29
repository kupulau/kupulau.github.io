---
title: Large Language Model
category: deep learning
tags:
  - NLP
url: https://velog.io/@kupulau/Large-Language-Model
created_at: 2024-12-06
related_notes:
---

**L**arge **L**anguage **M**odel(LLM; 대형 언어 모델)은 거대한 양의 사전 학습 텍스트 데이터로 훈련되고 수십억 개의 파라미터로 구성된, 범용적인 태스크 수행이 가능한 언어 모델이다. LLM의 예시로는 OpenAI의 ChatGPT, Meta의 LLaMA, MISTRAL AI의 Mistral, Upstage의 Solar 등이 있다. 
LLM 이전 단계인 pretrained language model (ex) GPT-1, GPT-2, BERT 등) 은 태스크별로 fine-tuning을 해서 목적별 모델을 구축해서 하나의 모델로 하나의 태스크를 해결하는 구조였다. 그러나 LLM은 사전 학습 및 fine-tuning을 통해 범용 목적의 모델을 구축해서 하나의 모델로 다양한 태스크를 해결할 수 있게 되었다. 


### LLM의 동작 원리
LLM은 zero-shot learning 및 few-shot learning을 기반으로 동작한다.

- Zero-shot learning : 모델이 prompt만으로 태스크를 이해하여 수행
- Few-shot learning : 모델이 prompt와 demonstration (예시)를 통해 태스크를 이해하여 수행

❓ prompt란
LLM에게 원하는 작업 및 실제 입력값을 제공하는 방식을 의미한다. prompt를 이용하여 모델의 태스크 수행 및 출력문을 제어한다. prompt는 instruction, demonstration, input으로 구성된다. 

- Instruction : 태스크 수행을 위한 구체적인 지시문
- Demonstration : 해당 태스크의 실제 예시 (입력-출력 쌍)
- Input : 실제 태스크 수행을 위한 입력문

> **prompt 예시**
>
\[Instruction] 위 예시는 영화 리뷰에 대한 분석 결과야. 예시를 보고 아래 리뷰의 감성을 분석해줘.
\[Demonstration] 
리뷰 : 러닝 타임 내내 웃음이 끊이지 않은 영화 
감성 : 긍정
\[Input] 
리뷰 : 너무 길고 지루했다.
감성 : 

<br>

### Architecture
LLM은 transformer 구조를 변형한 encoder-decoder와 decoder only 모델 구조를 사용한다.

#### Encoder-Decoder
![](https://velog.velcdn.com/images/kupulau/post/d8e33ce7-8f97-4583-8b03-afc5d4f887a8/image.png)

- encoder를 통해 입력을 이해하고, decoder를 통해 문장을 생성한다. 
- T5(**T**ext-**t**o-**T**ext **T**ranser **T**ransformer)가 encoder-decoder 구조로 된 언어 모델의 예시이다.
- Span corruption(텍스트 데이터를 모델이 학습할 때 사용하는 데이터 손상 기법)을 이용해 입력 문장을 이해하고 생성하는 능력을 학습하게 된다.
  - step 1. 입력 문장 중 임의의 span을 masking한다. 이 때 각 masking id를 부여한다.
  - step 2. masking된 문장을 encoder에 입력한다.
  - step 3. masking id와 복원 문장을 decoder에 입력한다.
  - step 4. decoder는 복원된 문장을 생성한다.
  
#### Decoder-only
![](https://velog.velcdn.com/images/kupulau/post/3875f7b7-53f5-43b5-8ffa-e243b377971a/image.png)

- 문장을 토큰 단위로 입력하여 매 토큰마다 다음 토큰을 예측하도록 학습한다.
- 대부분의 LLM이 causal decoder 구조를 사용

<br>

### Corpus
Corpus는 사전 학습을 위한 대량의 텍스트 데이터 집합이다. 원시 데이터에는 학습에 불필요한 데이터(욕설, 혐오 표현, 중복 데이터, 개인 정보 등)가 많이 포함되어 있으므로 이러한 데이터를 정제해서 corpus를 구축해야 한다. 

<br>

### Instruction tuning
Instruction tuning이란, 사용자의 광범위한 입력에 대해 안전하면서도 도움이 되는 적절한 답변을 하도록 fine-tuning하는 과정이다. instruction tuning은 Supervised Fine-Tuning, Reward Modeling, Reinforcement Learning with Human Feedback의 3단계로 구성된다.

#### Supervised Fine-Tuning (SFT)
prompt와 demonstration을 이용해 광범위한 사용자 입력에 대해 적절히 답변하도록 지도 학습을 한다. 

#### Reward modeling
ranking 기반 학습 방법을 사용하여 LLM의 생성문에 대한 인간의 선호도를 모델링한다. LLM의 생성문에 대해 helpfulness(질문의 의도에 맞는 유용한 정보를 답변으로 제공하는지?)와 safety(혐오/차별적 표현을 사용하거나 위험을 초래하는 문장이지는 않은지?)를 점수로 산출한다. 

#### Reinforcement Learning with Human Feedback (RLHF)
PRO 알고리즘을 이용하여 helpfulness와 safety를 만족하는 답변에 더 높은 점수를 부여하도록 학습한다.

이러한 과정을 통해 instruction tuning을 하면 사용자 지시 호응도가 상승하고 거짓 정보(hallucination)를 생성하는 빈도가 감소한다.

<br>

### 기존의 언어 모델 학습 방법론
기존에는 언어 모델을 target task에 맞게 **fine-tuning**하는 방법을 이용해 우수한 성능을 얻어 왔다. 이 방법들은 크게 feature-based approach, fine-tuning I, fine-tuning II의 3가지로 유형화 할 수 있다. 
feature-based approach는 사전학습 모델로부터 embedding을 추출하고 classifier를 학습하는 방법이다. Fine-tuning I과 Fine-tuning II는 각각 output layer를 업데이트, 모든 layer를 업데이트 하는 방법이다. 
성능은 기본적으로 더 많은 파라미터를 학습시킬 수록 향상되므로 많은 파라미터를 시키는 fine-tuning II, fine-tuning I, feature-based approach 순으로 성능이 높다. 다만 이의 역순으로 training efficiency는 떨어진다.

fine-tuning 외에도 **I**n-**C**ontext **L**earning (ICL)을 이용하여 언어 모델을 학습시켰다. ICL은 few-shot prompting으로, 새로운 가중치를 학습하거나 업데이트 하지 않고 몇 가지 예시를 prompt로 입력받아 새로운 태스크를 수행하는 것이다. 

그러나 fine-tuning 방법의 경우 모델을 지속적으로 추가 학습하는 과정에서 기존에 학습한 정보를 잊는다는 문제점이 발생한다. 그리고 각 downstream task마다 독립적으로 학습된 모델을 저장하고 배포할 때 막대한 시간과 컴퓨팅 자원이 필요하게 된다.
ICL의 경우에도, random한 label을 입력했을 때에도 문제를 잘 해결한다는 연구 결과가 존재해 ICL의 결과물을 신뢰하기 어렵다는 문제점이 있다. 

<br>

### PEFT
PEFT(**P**arameter-**E**fficient **F**ine-**T**uning)는 기존의 언어 모델 학습에서 파라미터를 많이 업데이트 시킬 때의 문제점을 해결하고자 하는 학습법이다. PEFT는 모델의 모든 파라미터를 학습시키지 않고 일부 파라미터만 fine-tuning하는 방법이다. 가장 대표적인 4가지 접근 방식은 Adapter tuning, Prefix tuning, Prompt tuning, Low-Rank Adaptation이다.

#### Adapter tuning
![](https://velog.velcdn.com/images/kupulau/post/b79c5f33-1f5a-432e-a485-2d0b6696ddcf/image.png)

기존에 이미 학습이 완료된 모델의 각 레이어에 학습 가능한 Feed Foward Network(FFN)를 삽입하는 구조이다. Adapter layer는 transformer의 vector를 더 작은 차원으로 압축한 후에 비선형 변환을 거쳐 원래 차원으로 복원하는 병목 구조로 이루어진다. 
Adapter 모듈은 finetuning 단계에서 특정 target task에 대해 최적화된다. 이 때 나머지 transformer layer는 모두 고정된다.
그러나 adapter tuning 방식은 병목 레이어를 추가함으로써 inference latency가 증가하여 실제로 사용하기 어렵다는 단점이 있다. 

```python
# Adapter tuning 구현 예시
def transformer_block_with_adapter(x):
	residual = x
    x = SelfAttention(x)
    x = FFN(x)     # Adapter
    x = LN(x + residual)
    residual = x
    x = FFN(x)     # transformer FFN
    x = FFN(x)     # Adapter
    x = LN(x + residual)
    return x
```

#### Prefix tuning
![](https://velog.velcdn.com/images/kupulau/post/49ddb97e-ac57-4f07-9ec4-45515c90d872/image.png)
transformer의 각 레이어에 prefix라는 훈련 가능한 vector를 추가하는 방법으로, prefix는 가상의 embedding으로 간주된다. 각 task를 더욱 잘 풀이하기 위한 벡터를 최적화하여 기존 모델과 병합할 수 있다. 

```python
# Prefix tuning 구현 예시
def transformer_block_for_prefix_tuning(x):
	soft_prompt = FFN(soft_prompt)
    x = concat([soft_prompt, x], dim=seq)
    return transformer_block(x)
```

#### Prompt tuning
모델의 입력 레이어에 훈련 가능한 prompt vector를 통합하는 방법이다. 이는 input 문장에 직접적인 자연어 prompt를 덧붙이는 prompting과는 다른 개념이며, embedding layer를 최적화하는 것이다. 이를 통해 target task에 최적화되는 튜닝을 할 수 있다.

```python
# Prompt tuning 구현 예시
def soft_prompted_model(input_ids):
	x = Embed(input_ids)
    x = concat([soft_prompt, x], dim=seq)
    return model(x)
```

#### Low-Ranks Adaptation (LoRA)
![](https://velog.velcdn.com/images/kupulau/post/fdf40522-36d8-446c-9f55-cdda5d601df7/image.png)

사전 학습된 모델의 파라미터를 고정하고 학습 가능한 rank decomposition 행렬을 삽입하는 방법이다. 이 행렬은 행렬의 차원을 rank 만큼 줄이는 행렬과, 다시 원래 차원 크기로 바꿔주는 행렬로 구성된다. 레이어마다 hidden states에 lora parameter를 더해서 tuning한다. 
LoRA를 이용하여 새롭게 학습한 파라미터를 기존 모델에 합쳐 줌으로써 추가 연산이 필요하지 않게 된다. 모델의 architecture를 변형하지 않고도 활용할 수 있다. 기존의 method들 대비 월등이 높은 성능을 보여 PEFT 방법들 중 가장 널리 쓰이는 방법이다. 

```python
def lora_linear(x):
	h = x @ W     # regular linear
    h += x @ W_A @ W_B     # low-rank update
    return scale * h
```

<br>

### LLM의 평가
LLM이 잘 동작하고 있는지는 어떻게 평가할 수 있을까? 다른 평가와의 큰 차이점은 "범용 태스크를 잘 수행하고 있는지"를 평가하는 것이다.
 
#### MMLU
MMLU(**M**assive **M**ultitask **L**anguage **U**nderstanding)는 LLM의 범용 태스크 수행 능력을 평가하기 위한 데이터 셋이다. 총 57개의 태스크로 구성되어 있으며, 생물, 정치, 수학, 물리학, 역사, 지리, 해부학 등의 다양한 분야를 다룬다. 객관식 형태로 평가를 진행하며, 정답 보기를 생성하면 문제를 맞춘 것으로 간주한다.

#### HellaSwag
HellaSwag는 LLM이 일반 상식을 보유하고 있는지 평가하기 위한 데이터 셋으로, 사람은 매우 쉽게 해결 가능한 태스크로 구성되어 있다. 주어진 문장에 이어질 자연스러운 문장을 선택하여 정답 보기를 생성하면 문제를 맞춘 것으로 간주한다.

#### HumanEval
HumanEval은 LLM의 코드 생성 능력을 평가하기 위한 데이터 셋이다. 함수명과 docstring(함수의 수행 과정과 의도하는 결과물을 명시하는 주석)을 입력받아 LLM이 생성한 코드로 생성한 결과물이 실제 값과 일치하면 문제를 맞춘 것으로 간주한다. 

#### LLM-Evaluation-Harness
LLM-Evaluation-Harness는 자동화된 LLM 평가 프레임워크로, MMLU, HellaSwag 등 다양한 평가 데이터 셋을 불러와 LLM을 평가할 수 있다. 

#### G-Eval
G-Eval은 LLM의 창의적 글쓰기 능력(자기소개서 수정, 광고 문구 생성, 어투 변경 등)을 평가하는 것이다. 기존에는 LLM이 생성한 문장을 인간이 정성적으로 품질을 평가했는데, 이 때문에 평가할 때의 비용이 크고 시간도 많이 소모되었다.
현재는 AutoCoT(모델 스스로 추론 단계를 구축하는 프롬프트 방식)을 통해 모델이 스스로 평가 단계를 정의하고 평가를 수행한 뒤 인간이 평가한 점수와의 correlation을 측정한다. 

<br>

### References
https://arxiv.org/pdf/2303.18223
https://arxiv.org/pdf/2101.00190
https://arxiv.org/pdf/2104.08691
https://arxiv.org/abs/2106.09685