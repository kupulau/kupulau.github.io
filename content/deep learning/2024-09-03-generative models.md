---
title: generative models
category: deep learning
tags:
  - genAI
url: https://velog.io/@kupulau/generative-models
created_at: 2024-09-03
related_notes:
---

생성 모델 (generative models) 은 분포에 맞는 데이터를 생성해내는 통계 모델이다.

### Autoregressive model
가장 간단한 generative model이다. 기본 아이디어는 $i$번째 pixel의 확률 $p(x_i)$가 이전까지의 픽셀들의 확률에 의해 결정되고, 이 확률 ($p(x)$; likelihood) 을 극대화시키는 방향으로 학습한다는 것이다. 

$p(x)  = \prod_{i=1}^n p(x_i|x_1, x_2, ... , x_{i-1})$

**pixelRNN**
autoregressive model의 한 예시로, 위 식의 conditional probability term을 RNN (LSTM 등) 으로 치환해서 표현한것이다. 
픽셀의 값을 순차적으로 결정하는데 이전의 픽셀들의 정보와 현재 픽셀의 정보를 조합하여 0 ~ 255 중 가장 probability가 높은 값을 픽셀값으로 채택하게 된다. 이전의 정보까지도 고려하기 때문에 느리다는 단점이 있다. 

<br>

### Variational AutoEncoder (VAE)
우선 **AutoEncoder**에서는 Input 데이터가 encoder를 통과하며 중요한 정보만 저장되고, bottleneck이라는 구조에 input 데이터를 저차원으로 매핑한 후, decoder에서 reconstruction을 하게 된다. 따라서 손실 함수도 input과 비교를 하는 reconstruction loss이다. 

AutoEncoder는 bottleneck에서 fixed vector를 매핑하지만, **VAE**에서는 bottleneck이 distribution이 되도록 학습을 진행한다. 
VAE의 기본 가정은 Latent factor $z$가 실제로 보이는 $x$를 샘플링하는데 기여를 한다는 것이다. (→ $p(x|z)$) 따라서 $p(x) = \int p(x|z)p(z) dz$ 의 likelihood를 도출해서 maximize한다. 그러나 이 식은 사실상 계산이 불가능하다. 따라서 posterior를 근사하여 해결한다. decoder에서 $p(x|z)$를 계산하고, encoder에서는 $p(z|x)$를 $z$의 분포를 잘 따르는 값으로 근사한다. 
그러나 출력되는 결과가 blurry하고 modeling capacity가 충분하지 않다는 단점이 있다.  

<br>

### Diffusion model
데이터가 주어지면 점점 gaussian noise를 더해주면서 (→ forward process) 최종적으로 gaussian distribution을 가지는 latent z로 매핑을 한 후 gaussian denoising을 하면서 (→ reverse process) 원본 이미지로 되돌아가는 방식이다. 

**Denoising Diffusion Probablistic Model (DDPM)** 은 diffusion model의 일종으로, 복잡한 데이터의 확률 분포를 학습하여 새로운 데이터를 생성하는 데 사용된다. 

그러나 노이즈를 계산하고 추가하고 denoising 하는 것은 시간이 매우 오래 걸리기 때문에, 이를 개선하기 위한 모델이 개발되었다.

<br>

### Latent diffusion (=Stable diffusion)

데이터를 encoder를 이용해 latent space로 매핑한다. Latent space에서 forward process를 거치면 nosie가 더해지며 gaussian noise에 가깝게 된다. Reverse process에서는 denoising을 진행한다. Reverse process는 U-Net 구조로 설계되어 있으며 condition을 cross attention 사이에 입력해줌으로써 condition을 생성할 수 있는 모델이 된다. 

<br>

### ControlNet
이미지 형태의 condition을 넣어주면 그 조건에 맞는 이미지를 생성한다.

<br>

### LoRA (Low-Rank Adaptation) 

<br>

### 활용 사례

**Prompt-to-prompt image editing with cross attention control**

Original image는 최대한 유지하면서 텍스트를 기반으로 한 이미지 수정 기법이다. Cross-attention map에 이미지의 geometry가 남아있기 때문에 이를 이용하면 전체적인 틀을 유지하면서도 변경사항이 텍스트로 주어졌을 때 그 부분만 수정할 수 있다.
단어를 바꾸어서 이미지를 수정하는 경우에는 Attention map 자체를 변경하고, 새로운 phrase를 추가하는 경우에는 attention map을 사이 사이에 삽입하는 식이다. 
그러나 이미지에 대한 full description을 잘 준비해야 한다는 단점이 있다.

<br>

**InstructPix2Pix**
prompt-to-prompt 방식의 단점을 보완하고자 나온 모델이라, full description이 필요하지 않다. 
Instruction-based image editing을 supervised learning으로 접근했다. 

<br>

**Marigold**
Stable diffusion 모델에 잘 학습된 이미지 knowledge를 depth estimation에 사용한 연구이다.
Latent encoder로 이미지를 latent화 ($z(x)$)하고, depth image도 latent화 한다 ($z(d)$). 그리고 latent화 한 depth image에 대해서 noise를 더한다. 그 후 latent화된 이미지와 concatenation을 하고 이를 통해서 이미지에 대해 conditional depth 계산할 수 있도록 한다. 그 후 latent diffusion U-net 을 통과하면서 denoising한다. denoising된 이미지는 latent decoder를 통과하며 output depth image가 나온다. 