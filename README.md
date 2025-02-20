# 2025-NLP

- [ ]  Words Representation
- [ ]  Sequence Modeling with RNNs
- [ ]  Sequence Modeling with Transformers
- [ ]  Instruction Tuning
- [ ]  Preference Optimization
- [ ]  Build NLP Projects with Hugging Face
- [ ]  Prompting
- [ ]  Parameter Efficient Fine-Tuning
- [ ]  Distillation, Quantization, Pruning
- [ ]  Mixture of Experts, Retrieval-Augmented Generation

### 📚 **NLP 개념 공부를 위한 추천 자료**

✅ **Words Representation (단어 표현)**

- 📖 *Speech and Language Processing* - Jurafsky & Martin (챕터 6: 단어 표현)
- 📝 Word2Vec, GloVe 관련 논문:
    - "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    - "GloVe: Global Vectors for Word Representation" (Pennington et al.)
- 🏫 [CS224N Stanford NLP 강의 - Word Vectors](https://www.youtube.com/watch?v=ERibwqs9p38)

✅ **Sequence Modeling with RNNs**

- 📖 *Dive into Deep Learning* (챕터 9: 순환 신경망)
- 🏫 [CS231n RNN 강의](https://www.youtube.com/watch?v=9zhrxE5PQgY)
- 💻 PyTorch 공식 튜토리얼: RNN을 이용한 NLP

✅ **Sequence Modeling with Transformers**

- 📖 *Attention Is All You Need* 논문 (Vaswani et al.)
- 🏫 [Stanford CS224N - Transformers](https://www.youtube.com/watch?v=iDulhoQ2pro)
- 💻 PyTorch 공식 Transformer 튜토리얼: Transformer 모델 구현

✅ **Instruction Tuning & Preference Optimization**

- 📖 InstructGPT 논문: ["Training language models to follow instructions"](https://arxiv.org/abs/2203.02155)
- 📖 RLHF (Reinforcement Learning from Human Feedback) 논문: ["Deep Reinforcement Learning from Human Preferences"](https://arxiv.org/abs/1706.03741)
- 💻 OpenAI의 RLHF 튜토리얼: [Reinforcement Learning with Human Feedback](https://openai.com/research/instruction-following)

✅ **Build NLP Projects with Hugging Face**

- 📖 *Natural Language Processing with Transformers* (Lewis Tunstall, Leandro von Werra, Thomas Wolf)
- 🏫 Hugging Face 공식 코스
- 💻 실습: Hugging Face `transformers` 라이브러리 활용하여 BERT, GPT 모델 활용

✅ **Prompting & Parameter Efficient Fine-Tuning**

- 📖 [Prompt Engineering Guide](https://www.promptingguide.ai/)
- 🏫 LoRA, Adapter 관련 논문:
    - ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
    - ["Adapters: Efficient Transfer Learning"](https://arxiv.org/abs/1902.00751)
- 💻 실습: Hugging Face `peft` 라이브러리 활용하여 LoRA 적용

✅ **Distillation, Quantization, Pruning**

- 📖 *Model Compression Techniques* 논문
- 🏫 [DistilBERT 논문](https://arxiv.org/abs/1910.01108)
- 💻 실습:
    - `torch.quantization`을 사용한 모델 경량화
    - `distilbert` 및 `TinyBERT`를 활용한 경량 NLP 모델 실험

✅ **Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG)**

- 📖 ["GLaM: Efficient Scaling of Large Language Models"](https://arxiv.org/abs/2112.06905)
- 📖 ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401)
- 💻 실습:
    - Hugging Face `rag-token`을 활용한 RAG 모델 구현
    - `Switch Transformer`나 `GLaM` 논문 구현 코드 분석
 

Lab 1: Minimal Byte Pair Encoding (10pt)

Lab 3: Nano GPT (10pt)

Lab 4: Building Your NLP Projects (10pt)

Lab 5: Parameter Efficient Fine-Tuning (10pt)

### 🛠 **Lab 실습 방법 가이드**

### **Lab 1: Minimal Byte Pair Encoding (BPE)**

- 목표: BPE 알고리즘을 직접 구현하여 토큰화를 실험
- 실습:
    - 작은 데이터셋에서 BPE 알고리즘 구현 (Python, NumPy 활용)
    - `sentencepiece` 및 `Hugging Face tokenizers`를 이용한 BPE 적용

### **Lab 3: NanoGPT**

- 목표: 간단한 GPT 모델을 직접 학습시키고 동작 방식 이해
- 실습:
    - `karpathy/nanoGPT` 코드 분석 및 실행
    - 작은 데이터셋 (예: Shakespeare text)으로 GPT 학습 실험
    - 하이퍼파라미터 튜닝 실습 (토큰 크기, 레이어 수 조정 등)

### **Lab 4: Building Your NLP Projects**

- 목표: NLP 프로젝트 기획 및 데이터셋 구축
- 실습:
    - Hugging Face `datasets` 라이브러리를 활용해 데이터셋 정제
    - `transformers`를 활용하여 사전 학습된 모델 적용
    - 특정 NLP 태스크 (예: 감성 분석, 요약) 수행

### **Lab 5: Parameter Efficient Fine-Tuning (PEFT)**

- 목표: LoRA, Adapter 등을 사용한 효율적인 모델 튜닝
- 실습:
    - Hugging Face `peft` 라이브러리 활용
    - 기존 GPT 모델에 LoRA 적용 후 성능 비교
    - 데이터셋 크기별 Fine-tuning 실험
