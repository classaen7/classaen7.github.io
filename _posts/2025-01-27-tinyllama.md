---
layout: post
title: '[논문 리뷰] TinyLlama: An Open-Source Small Language Model'
excerpt: SLM 모델 TinyLlama 논문 리뷰

categories:
  - 논문 리뷰
tags:
  - [논문 리뷰, SLM, TinyLlama]


date: 2025-01-27 19:24 +0900


---


![TinyLlama_logo](/assets/posts/TinyLlama/TinyLlama_logo.png){: style="display: block; margin: auto; width: 50%;"}


LLM(Large Language Model)을 학습하거나 사용할 때 하드웨어적 제약으로 인해 개인 컴퓨터나 GPU 서버에서 모델을 실행하기 어려운 경우가 많다. 이러한 한계는 작은 언어 모델(SLM)을 찾게 되는 주요 동기가 된다.

TinyLlama는 SLM(Small Language Model) 중 하나로, 기존 Llama 모델을 작게 만들어 효율적으로 만든 모델이다. TinyLlama 논문을 읽고 어떤 방식으로 SLM을 구현했는지를 확인해보겠다.



## Abstract

해당 논문은 TinyLlama 모델의 다음의 3가지 내용을 시사한다.

- **Llama 2 기반의 1.1B 규모의 작은 언어 모델** <br>

- **FlashAttention과 Lit-GPT 등의 기술을 활용한 계산 효율성 향상** <br>

- **다양한 다운스트림 작업에서 뛰어난 성능** <br>



## 1. Introduction

자연어 처리(NLP)에서 언어 모델의 성능 향상은 주로 모델 크기와 학습 데이터의 확장에서 비롯되며, 일부 연구는 대규모 모델에서만 나타나는 능력<font color="gray">(emergent intelligence)</font>을 발견했다.  <br>
그러나 작은 모델을 더 많은 데이터로 학습시키는 가능성은 충분히 탐구되지 않았다. 
이 논문은 스케일링 법칙의 한계를 넘어, 10억 개 매개변수를 가진 Llama2 모델을 대규모 데이터로 학습시켜 작은 모델의 성능과 가능성을 탐구한다.



## 2. Pre-training

TinyLlama의 Pre-training 과정에 대한 데이터셋, 모델 아키텍처, 하이퍼파라미터를 다룬다.



### 2.1 Pre-training data

**SlimPajama**와 **Starcoderdata** 데이터를 약 7:3 비율로 결합하여 9500억 개의 토큰을 생성한다. 

이때 SlimPajam는 RedPajama 데이터셋에서 파생된 고품질 코퍼스를 포함하며 Starcoderdata는 Github의 이슈와 텍스트-코드 쌍을 포함한다. 

> Github에서의 여러 TinyLlama 모델들의 데이터 수는 105B ~ 3T까지 다양함을 확인할 수 있다. 집중해야할 것은 Llama2와 같은 데이터 크기인 2T 토큰수의 데이터도 스케일 법칙의 한계를 넘어서 좋은 성능을 얻었다는 것에 초점을 맞춘다.

<br>

- **데이터 예시 : Starcoderdata**

|max_stars_repo_path|max_stars_repo_name|content|
| :----: | :---: | :--- |
|source/rule_lists.ads|jquorning/CELLE|with Ada.Containers. Doubly_Linked_Lists; <br>  limited with Rules;<br>  package Rule_Lists is type Rule_Access is access all Rules.Rule_Record; <br>  package Lists is new Ada.Containers.Doubly_Linked_Lists (Element_Type => Rule_Access); <br>  -- function Element (List : Lists.List) return Natural;  end Rule_Lists; |


> 각 데이터셋의 자세한 예시는 허깅페이스를 통해 확인 가능하다.

### 2.2 Architecture

Llama 모델 시리즈의 아키텍처를 따르며, Transformer 디코더만을 사용한다.


- **HyperParameters**

| Hidden Size | Intermediate Hidden Size | Context Len | Heads | Layers | Vocab Size |
|-------------|---------------------------|-------------|-------|--------|------------|
| 2,048       | 5,632                     | 2,048       | 32    | 22     | 32,000     |


- **Positional Embedding**

Rotary Positional Embedding (RoPE)을 사용하여 위치 정보를 모델에 삽입한다. 
<br>

- **Pre-norm and RMSNorm**

Transformer 서브 레이어 입력을 정규화(Pre-norm)하여 학습 안정성 개선한다.
효율성을 높이기 위해 RMSNorm 정규화 함수를 적용한다.
<br>

- **SwiGLU**

Swish 활성화 함수와 Gated Linear Units (GLU)을 결합한 활성화 함수를 사용한다.
<br>

- **Grouped-query Attention**

![TinyLlama_logo]( /assets/posts/TinyLlama/gqa.png){: style="display: block; margin: auto; width: 100%;"}

Transformer 모델에서 메모리 효율성을 높이고 추론 속도를 개선하기 위해 설계된 Attention 메커니즘의 변형이다.



### 2.3 Speed Optimization


#### Fully Sharded Data Parallel (FSDP)

```

![TinyLlama_logo]( /assets/posts/TinyLlama/fsdp.png){: style="display: block; margin: auto; width: 100%;"}

학습 과정에서 FSDP를 통합하여 다중 GPU 및 다중 노드 환경을 효과적으로 활용한다. 이를 통해 학습 속도와 효율성을 크게 개선하였다.


#### FlashAttention
FlashAttention-2를 통합하여 최적화된 어텐션 메커니즘을 구현한다. 이와 함께 Fused LayerNorm, Fused Cross Entropy Loss, Fused Rotary Positional Embedding 등을 사용하여 계산 처리량을 크게 향상시켰다.


#### xFormers
기존 SwiGLU 모듈을 xFormers의 Fused SwiGLU 모듈로 교체하여 효율성을 더욱 강화하였다.

```

#### Speed Comparison with Existing Models

|  | GPU Hours|
| :--: | :--:|
| Pythia-1.0B | 4,830 |
|MPT-1.3B | 7,920|
| TinyLlama| 3,456|


속도 향상 모듈을 구현한 결과, A100-40G GPU에서 초당 24,000개의 토큰 처리량을 달성했습니다. Pythia-1.0B 및 MPT-1.3B와 비교하여 GPU 시간 측면에서 더 효율적인 학습 속도를 기록

### 2.4 Training & 2.5 Version 1.1

![TinyLlama_logo]( /assets/posts/TinyLlama/figure_1.png){: style="display: block; margin: auto; width: 100%;"}

TinyLlama는 다음 토큰을 예측하는 방식인 Autoregressive 방식으로, 다음의 설정을 기반으로 TinyLlama는 효율적으로 학습을 진행하였다.

| 항목                  | 설정값                         |
|-----------------------|--------------------------------|
| Optimizer             | AdamW                         |
| β₁, β₂                | 0.9, 0.95                           |
| Weight Decay          | 0.1                           |
| 최대 학습률           | \( 4.0 \times 10^{-4} \)       |
| 최소 학습률           | \( 4.0 \times 10^{-5} \)       |
| 워밍업 단계           | 2,000 스텝                    |
| 배치 크기             | 200만 토큰                    |
| Gradient Clipping     | \( 1.0 \)                     |
| GPU                   | 16개의 A100-40G               |


### TinyLlama v1.1

스케줄러와 데이터 로딩 과정에 관한 implementation 이슈를 해결하기 위해 모델을 처음부터 다시 학습시켰고 다음과 같은 주요 변경 사항과 개선이 이루어졌다.


 
#### 1. 통신 오버헤드 감소
FSDP(Fully Sharded Data Parallel)를 활용해 노드 내부에서만 모델 파라미터를 샤딩하도록 변경하여 통신 오버헤드를 줄였


#### 2. 학습 데이터 양 감소
학습 데이터의 총 토큰 수를 3조 개에서 2조 개로 줄였습니다

데이터 양을 줄였음에도 불구하고 다운스트림 작업에서 성능이 약간 개선된 것을 관찰했습니다(자세한 내용은 섹션 3 참조).


#### 3. 3단계 Pre-training 과정
단일 사전 학습 프로세스 대신, 최신 연구(Wei et al., 2023)를 참고하여 3단계 학습 과정을 도입

1. Basic Pre-training
SlimPajama(Soboleva et al., 2023) 데이터만으로 일반적인 상식 추론 능력을 학습.

약 1.5조 개의 토큰을 학습하여 이후의 전문화된 학습을 위한 기반을 구축



2. Continual Pre-training : 특정 도메인 학습

SlimPajama와 함께 다양한 도메인 데이터를 활용

수학 및 코딩 데이터를 포함하는 Starcoder와 Proof Pile 2, 그리고 중국어 데이터를 포함하는 Skypile을 도입.

- TinyLlama v1.1: 일반적인 작업을 위한 모델.
- TinyLlama v1.1 - Math&Code: 수학 및 코딩 작업에 특화.
- TinyLlama v1.1 - Chinese: 중국어 텍스트 처리에 최적화.

3. Cooldown Phase

학습 과정 마지막 단계에서 배치 크기를 180만 토큰에서 720만 토큰으로 증가시켜 수렴성을 개선.

코사인 학습률 스케줄을 유지하며, 추가로 1500억 개의 토큰으로 학습.


## 3 Results

기존 오픈소스 모델인 디코더 전용 아키텍처를 사용하는 약 10억 개의 매개변수를 가진 언어 모델인 OPT-1.3B, Pythia-1.0B, Pythia-1.4B와 비교

다음 3가지 태스크에 대해서 위의 다른 모델들과 비교했을 때 좋은 결과를 얻었다. 기존 모델들보다 뛰어난 성능을 보였다.

1. Commonsense reasoning tasks
2. Problem-solving tasks
3. Evaluation for Chinese tasks


> 일부 모델 (TinyLlama v1.0 및 v1.1 Math&Code)는 중국어 데이터가 포함되지 않았음에도 불구하고 성능 향상이 관찰되었는데, 이는 학습 데이터 일부에서 중국어가 포함된 코딩 데이터의 영향을 받은 것으로 분석됩니다.


## 4 Conclusion
컴팩트한 구조와 우수한 성능을 갖춘 TinyLlama는 모바일 기기에서의 최종 사용자 응용 프로그램 구현을 가능하게 하며, 언어 모델과 관련된 혁신적인 아이디어를 테스트할 수 있는 경량 플랫폼으로 활용될 수 있습니다.
우리는 또한 다단계 학습과 데이터 스케줄링의 효과를 TinyLlama v1.1 시리즈를 통해 추가적으로 검증했습니다. 오픈소스 언어 모델 사전 학습 커뮤니티의 투명성을 촉진하기 위해, 우리의 사전 학습 코드, 중간 모델 체크포인트, 데이터 처리 단계에 대한 세부 정보를 모두 공개했습니다.

TinyLlama가 언어 모델 연구를 민주화하고, 이 분야의 미래 연구를 위한 유용한 통찰을 제공하는 데 중요한 기여를 할 것으로 믿습니다.

> 오픈소스로 확인가능하다는 것이 큰 장점이 되는 것 같다

## 개인 해석
Paper with Codes에서 SLM 모델에 대해 찾아보면 가장 상단에 있는 논문인 TinyLlama 모델에 대해서 리뷰해보았다.

LLM의 성능을 아직까지 왜 잘되는지에 대한 정답을 찾지 못했다는 점에서 모델이 스케일 법칙의 한계를 어떻게 이겨냈는지 와닿지는 않는다. 단순히 결과론적으로 작은 크기의 모델에도 큰 데이터셋이 좋은 성능을 일구는데 도움이 된다는 것에 초점을 맞춘 논문인것 같다.

기대했던 것은 LLM모델을 어떻게 압축하여 SLM으로 만들었을까라는 궁금증에서 해당 논문을 리뷰하게 되었는데, 그런것 보다는 모델을 더 효율적으로 학습시키는 기술들에 대한 공부의 필요성을 느끼면서 논문 리뷰를 마무리 하겠다.



## Further Task

스케일링 법칙 논문

- [ ] ROPE

- [X] Grouped-query Attention

FlashAttention

Llama2

Fully Sharded Data Parallel (FSDP)



