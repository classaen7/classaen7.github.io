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


![TinyLlama_logo](/assets/img/TinyLlama_logo.png){: style="display: block; margin: auto; width: 50%;"}


LLM(Large Language Model)을 학습하거나 사용할 때 하드웨어적 제약으로 인해 개인 컴퓨터나 GPU 서버에서 모델을 실행하기 어려운 경우가 많다. 이러한 한계는 작은 언어 모델(SLM)을 찾게 되는 주요 동기가 된다.

TinyLlama는 SLM(Small Language Model) 중 하나로, 기존 Llama 모델을 작게 만들어 효율적으로 변환한 모델이다. TinyLlama 논문을 읽고 각 목차의 중요 내용을 탐구하겠다.



# Abstract

- 1.1B 규모의 작은 언어 모델

- Llama 2의 아키텍처와 토크나이저를 기반

- FlashAttention과 Lit-GPT와 같은 오픈소스 커뮤니티의 여러 발전된 기술을 활용하여 계산 효율성을 크게 향상

- 다양한 다운스트림 작업에서 뛰어난 성능

- GitHub에 공개

# Introudce

대규모 언어 모델(LLM)은 방대한 텍스트 데이터로 사전 학습되어 다양한 작업에서 효과를 입증

일부 연구에서는 모델의 파라미터 수가 충분히 클 때만 나타나는 능력(예: Few-shot 학습 및 Chain-of-thought 추론)이 존재함을 보였다 - imergent itelligence

LLM의 확장 법칙(scaling behavior) - 최적의 계산 모델을 학습하기 위해 모델 크기와 학습 데이터 양을 비례적으로 늘려야 한다고 제안

작은 모델을 더 많은 데이터로 학습시키는 가능성은 충분히 탐구되지 않았다 -> 확장 법칙을 초과하는 데이터량으로 작은 모델을 학습하는 접근법을 탐구


# 2. Pre-training

## 2.1 Pre-training data

두 데이터셋을 결합하여 약 9500억 개의 토큰을 생성하고, Llama 토크나이저를 사용해 처리

### SlimPajama
`RedPajama` 데이터셋에서 파생된 고품질 코퍼스.

중복 제거 및 데이터 정제를 통해 원래 RedPajama의 50% 토큰만 유지.

Llama의 사전 학습 데이터와 유사한 구조.

TinyLlama는 이 데이터를 약 3에포크 동안 학습했으며, 총 3조 개의 토큰을 처리했다. SlimPajama와 StarCoder 데이터의 샘플링 비율은 약 7:3으로 설정

### StarCoder Training Dataset

86개의 프로그래밍 언어에 대한 코드 데이터를 포함하며, GitHub 이슈와 텍스트-코드 쌍도 포함.

SlimPajama에도 GitHub 데이터가 포함되어 있어 중복을 방지하기 위해 SlimPajama에서 GitHub 데이터를 제거하고 StarCoder의 코드 데이터만 샘플링.


## 2.2 Architecture

전체적인 architecture의 경우 Llama 모델 시리즈(Llama 2 포함)의 아키텍처를 따르며, Transformer 디코더만 사용한다.
아키텍처에서 모델의 파라미터를 줄이거나 무게를 가볍게하는 기법이 따로 있을 거라고 생각했는데 아키텍처 구조 자체는 딱히 그런건 없는 것 같다.

### Positional embedding


### Pre-norm and RMSNorm

### SwiGLU

### Grouped-query Attention


## 2.3 Speed Optimization

사실 해당 논문에서 모델의 크기를 줄이는 핵심 기법은 여기인것 같다.


### Fully Sharded Data Parallel (FSDP)
다중 GPU와 다중 노드 환경에서 학습 속도를 높임


### FlashAttention
최적화된 Attention 메커니즘으로, 계산 처리량을 향상.


### xFormers
메모리 사용량을 줄이고 더 큰 배치 크기를 허용하여 학습 효율 증가


- Speed Comparison with Existing Models


## 2.4 Training

