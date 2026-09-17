# 📘 Complete BERT Fine-Tuning Guide (In-Depth Notes in Hinglish)

Yeh document **BERT (Bidirectional Encoder Representations from Transformers)** ke fine-tuning ka complete conceptual aur practical guide hai. Isme theoretical concepts, mathematical intuitions, hyperparameters, aur 4 alag-alag NLP tasks ke implementations ko aasan bhasha (Hinglish) me explain kiya gaya hai.

---

## 📑 Table of Contents
1. [BERT Kya Hai aur Kaise Kaam Karta Hai?](#1-bert-kya-hai-aur-kaise-kaam-karta-hai)
2. [Pre-training vs Fine-Tuning](#2-pre-training-vs-fine-tuning)
3. [BERT Tokenizer aur Input Representation](#3-bert-tokenizer-aur-input-representation)
4. [Model Loading: UNEXPECTED aur MISSING Weights ka Secret](#4-model-loading-unexpected-aur-missing-weights-ka-secret)
5. [Sabhi Hyperparameters ka In-Depth Breakdown](#5-sabhi-hyperparameters-ka-in-depth-breakdown)
6. [Hugging Face Trainer vs Custom PyTorch Loop](#6-hugging-face-trainer-vs-custom-pytorch-loop)
7. [BERT for Different NLP Tasks (Deep Dive)](#7-bert-for-different-nlp-tasks-deep-dive)
   - Task 1: Single Sentence Classification (IMDB Sentiment)
   - Task 2: Token Classification / Named Entity Recognition (NER)
   - Task 3: Extractive Question Answering (SQuAD Style)
   - Task 4: Sentence Pair Classification / NLI
8. [Common Errors aur Unke Solutions (NCCL, 403 Forbidden, etc.)](#8-common-errors-aur-unke-solutions)
9. [Quick Revision Cheat Sheet](#9-quick-revision-cheat-sheet)

---

## 1. BERT Kya Hai aur Kaise Kaam Karta Hai?

### Core Concept:
- **BERT** ka full form hai: **Bidirectional Encoder Representations from Transformers**.
- Google ne ise 2018 me introduce kiya tha.
- BERT **Transformer ke Encoder** architecture par based hai (decoder isme nahi hota).
- **Bidirectional ka matlab**:
  - Traditional models (jaise GPT ya LSTMs) left-to-right padhte hain.
  - BERT ek sentence ke har word ko uske **Left aur Right dono contexts** se ek sath attend karta hai.
  - *Example*: Sentence hai `"Bank of the river"` vs `"Bank of America"`. GPT "Bank" padhte waqt agle words nahi dekh sakta, par BERT dono contexts ko dekh kar samajh leta hai ki pehla "Bank" kinara hai aur doosra financial institution.

### BERT Base vs BERT Large Architecture:
| Spec | BERT Base | BERT Large |
| :--- | :---: | :---: |
| **Layers ($L$)** | 12 Transformer Blocks | 24 Transformer Blocks |
| **Hidden Size ($H$)** | 768 | 1024 |
| **Attention Heads ($A$)** | 12 | 16 |
| **Total Parameters** | ~110 Million | ~340 Million |

---

## 2. Pre-training vs Fine-Tuning

### Pre-training (Massive Unsupervised Learning):
BERT ko do main objectives par train kiya gaya tha (Wikipedia + BooksCorpus, ~3.3 Billion words):
1. **Masked Language Modeling (MLM)**: Sentence me se random 15% words ko `[MASK]` kar diya jata hai, aur model ko unhe predict karna hota hai.
2. **Next Sentence Prediction (NSP)**: Model ko do sentences (A aur B) diye jate hain aur batana hota hai ki kya B sentence A ke turant baad aata hai (`IsNext`) ya random hai (`NotNext`).

### Fine-Tuning (Supervised Task-Specific Transfer Learning):
- Pre-training me BERT ne English language ki grammar, syntax, aur semantics seekh li.
- Fine-tuning me hum pretrained weights ko freeze ya low learning rate ke sath downstream tasks (Sentiment analysis, NER, QA, etc.) ke liye adapt karte hain.
- **Faayda**: Hame scratch se billion tokens train karne ki zaroorat nahi padti; sirf kuch hazar samples aur kuch minutes me state-of-the-art results mil jate hain.

---

## 3. BERT Tokenizer aur Input Representation

### WordPiece Tokenizer:
BERT character-level ya pure word-level use nahi karta, balki **WordPiece** subword tokenization use karta hai (Vocab size = 30,522):
- Frequent words pure rehte hain (jaise `"cat"`, `"movie"`).
- Rare ya complex words subwords me toot jate hain jinke aage `##` lagta hai:
  - `"unbelievable"` $\rightarrow$ `['un', '##belie', '##vable']`
  - Faayda: Out-of-Vocabulary (OOV) problem bilkul khatam ho jati hai.

### Special Tokens:
- **`[CLS]` (Token ID 101)**: Har input sequence ke starting me lagta hai. Iska final hidden state poore sentence ka summary representation hota hai jo classification me use hota hai.
- **`[SEP]` (Token ID 102)**: Sentence ke end me ya do sentences ko separate karne ke liye lagta hai.
- **`[PAD]` (Token ID 0)**: Short sentences ko `max_length` tak pad karne ke liye use hota hai.
- **`[MASK]` (Token ID 103)**: Pre-training me masked words ke liye.

### 3 Embeddings ka Sum:
Jab koi input sequence BERT me jata hai, to har token ka final vector 3 embeddings ko add karke banta hai:
$$\text{Input Embedding} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$$

1. **Token Embedding**: WordPiece token ka vector ($768$-dim).
2. **Segment (Token Type) Embedding**: Sentence A ke tokens ke liye `0`, Sentence B ke tokens ke liye `1`.
3. **Position Embedding**: Token ki sequence me position ($0, 1, 2, \dots, 511$). (Kyuki Self-Attention me order inherently nahi hota).

### Attention Mask:
- Attention mask BERT ko batata hai ki kin tokens par dhyan dena hai aur kinhe ignore karna hai:
  - Real tokens $\rightarrow$ `1`
  - `[PAD]` tokens $\rightarrow$ `0`
- Padding tokens par self-attention calculate nahi hota, jisse computation waste nahi hoti aur padding loss ko corrupt nahi karti.

---

## 4. Model Loading: UNEXPECTED aur MISSING Weights ka Secret

Jab aap likhte hain:
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```
Console me warning aati hai:
- `UNEXPECTED`: `cls.predictions.transform.dense...`, `cls.seq_relationship...`
- `MISSING`: `classifier.weight`, `classifier.bias`

### Iska Matlab Kya Hai?
- **Pretrained BERT** ke pass pre-training ke do heads the: Masked LM head aur Next Sentence Prediction head.
- Jab humne `BertForSequenceClassification` call kiya:
  1. Hugging Face ne purane MLM/NSP heads ko delete kar diya (isliye woh `UNEXPECTED` report hote hain kyuki nayi class me unki jagah nahi hai).
  2. Ek brand new randomly initialized Linear Layer (`Linear(768, 2)`) add kar di classification ke liye (isliye woh `MISSING` report hoti hai kyuki checkpoint me iske weights nahi the).
- **Conclusion**: Yeh bilkul normal aur expected behavior hai!

---

## 5. Sabhi Hyperparameters ka In-Depth Breakdown

Fine-tuning me hyperparameters ka selection sabse important hota hai:

### 1. `learning_rate` (Recommended: `2e-5` to `5e-5`)
- **Kyun itna kam?** Scratch se training me learning rate $10^{-3}$ ya $10^{-4}$ hota hai. Lekin BERT pehle se trained hai. Agar aap bada learning rate (jaise $10^{-3}$) use karoge, to **Catastrophic Forgetting** ho jayegi (BERT apni purani saari knowledge bhool jayega aur weights corrupt ho jayenge).
- $2 \times 10^{-5}$ ($0.00002$) weights ko gently fine-tune karta hai.

### 2. `per_device_train_batch_size` (Recommended: `8`, `16`, ya `32`)
- GPU VRAM aur gradient stability ke beech balance.
- Batch size 8 ya 16 standard GPUs ke liye perfect hai. Batch size jitna bada hoga, gradient descent utna smooth hoga, par memory utni zyada lagegi.

### 3. `num_train_epochs` (Recommended: `2` to `4`)
- BERT downstream tasks par bohot tezi se converge hota hai.
- 2 se 3 epochs me 90%+ accuracy aa jati hai. 5 se zyada epochs train karne par model small datasets par severely **overfit** ho jata hai.

### 4. `weight_decay` (Recommended: `0.01`)
- Yeh **L2 Regularization** hai jo weights ko bohot bada hone se rokta hai:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{loss}} + \lambda \sum w^2$$
- Best practice: Weight decay ko linear weights par lagaya jata hai, par `bias` aur `LayerNorm.weight` par `0.0` rakha jata hai.

### 5. `num_warmup_steps` / `warmup_ratio` (Recommended: `10%` of total steps)
- **Concept**: Training ke shuruati 10% steps me Learning Rate $0$ se linearly badh kar peak value ($2 \times 10^{-5}$) tak jata hai, aur bache hue 90% steps me gradually zero tak decay hota hai.
- **Kyun zaroori hai?** Shuruat me classification head ke weights bilkul random hote hain. Agar pehle hi step par poora learning rate de diya jaye, to random head se aane wale massive gradients pretrained BERT layers ko destabilize kar denge. Warmup head ko pehle settle hone deta hai.

### 6. `max_norm` (Gradient Clipping, Recommended: `1.0`)
- Backpropagation ke time agar gradients ka vector norm $1.0$ se bada ho jata hai, to PyTorch use rescale kar deta hai:
  $$\mathbf{g} \leftarrow \mathbf{g} \times \frac{\text{max\_norm}}{\max(\|\mathbf{g}\|, \text{max\_norm})}$$
- Yeh 12-layer deep transformer me **exploding gradients** ko prevent karta hai.

### 7. `max_length` (Recommended: `128` ya `256`)
- Transformer Self-Attention ki time aur memory complexity hoti hai:
  $$\mathcal{O}(N^2)$$
- Agar sequence length 512 se ghata kar 256 kar di jaye, to memory aur speed me **$4\times$ improvement** aata hai! Isliye agar texts chote hain, to `max_length=256` ya `128` rakhna best practice hai.

---

## 6. Hugging Face Trainer vs Custom PyTorch Loop

| Feature | Hugging Face `Trainer` | Custom PyTorch Loop |
| :--- | :--- | :--- |
| **Ease of Use** | Bohot easy, 10 lines me complete pipeline. | Manual code likhna padta hai (DataLoader, loops). |
| **Built-in Features** | Auto logging, evaluation, checkpointing, mixed precision (`fp16/bf16`). | Sab kuch manually track aur implement karna hota hai. |
| **Customizability** | Deep custom losses ya complex architecture me thoda rigid. | $100\%$ control har forward, backward aur gradient step par. |
| **When to use?** | Standard tasks (Classification, NER, QA) ke fast production runs ke liye. | Research, custom loss functions, ya deep understanding ke liye. |

---

## 7. BERT for Different NLP Tasks (Deep Dive)

BERT ek versatile model hai. Head badal kar hum ise alag-alag tasks ke liye use karte hain:

```
                      +-----------------------------+
                      |         BERT Base           |
                      |  (12 Layers Transformer)    |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
[Task 1: Sequence Classif.]  [Task 2: Token Classif.]   [Task 3: Question Answering]
         |                           |                           |
Pool [CLS] -> Linear(768, C)  Every Token -> Linear(768, K)  Every Token -> Start/End Logits
         |                           |                           |
Output: Single Label (0/1)    Output: Tag per token       Output: Answer Span [start:end]
```

---

### Task 1: Single Sentence Classification (Sentiment Analysis)
- **Goal**: Poore sentence ka sentiment (Positive=1 / Negative=0) ya topic find karna.
- **Model Class**: `BertForSequenceClassification`
- **Internal Working**:
  - Sentence: `[CLS] The movie was fantastic! [SEP]`
  - BERT encoder 12 layers ke baad har token ka 768-dim vector deta hai.
  - Classification head sirf pehle token **`[CLS]`** ke representation ko leta hai.
  - Linear layer: `Linear(in_features=768, out_features=num_classes)`.
  - Softmax probability se highest score wala class predict hota hai.

---

### Task 2: Token Classification / Named Entity Recognition (NER)
- **Goal**: Sentence ke har word ko ek label dena (Person, Organization, Location, Other).
  - *Example*: `"Elon Musk leads Tesla in Texas"`
  - `"Elon"` $\rightarrow$ `B-PER`, `"Musk"` $\rightarrow$ `I-PER`, `"Tesla"` $\rightarrow$ `B-ORG`, `"Texas"` $\rightarrow$ `B-LOC`.
- **Model Class**: `BertForTokenClassification`
- **The Critical Subword Problem & Solution**:
  - Pre-tokenized word hai: `"HuggingFace"` (Label: `B-ORG`).
  - WordPiece tokenizer ise 3 subwords me tod deta hai: `['Hu', '##gging', 'Face']`.
  - Ab labels kaise assign karein?
  - **Rule**:
    1. Pehle subword (`'Hu'`) ko original label (`B-ORG`) do.
    2. Baaki subwords (`'##gging'`, `'Face'`) aur special tokens (`[CLS]`, `[SEP]`, `[PAD]`) ko label **`-100`** do.
  - PyTorch ka `CrossEntropyLoss(ignore_index=-100)` by default `-100` ko ignore karta hai, isliye gradients sirf accurate first subword par calculate hote hain!

---

### Task 3: Extractive Question Answering (SQuAD Style)
- **Goal**: Ek Paragraph (Context) aur Question diya hota hai, model ko paragraph ke andar se answer ka starting aur ending word dhoondhna hota hai.
- **Model Class**: `BertForQuestionAnswering`
- **Input Format**:
  `[CLS] What is BERT? [SEP] BERT is a bidirectional transformer model. [SEP]`
- **Internal Working**:
  - QA classification head har token ke vector ko **2 scalar outputs** me project karta hai:
    1. `start_logits`: Probability ki answer is token se shuru ho raha hai.
    2. `end_logits`: Probability ki answer is token par khatam ho raha hai.
  - Prediction:
    $$\text{start\_index} = \text{argmax}(\text{start\_logits})$$
    $$\text{end\_index} = \text{argmax}(\text{end\_logits})$$
  - Context me se `tokens[start_index : end_index + 1]` slice karke text decode kar liya jata hai.

---

### Task 4: Sentence Pair Classification / Natural Language Inference (NLI)
- **Goal**: Do sentences ke beech ka logical relation find karna:
  - **Entailment** (Sentence A implies Sentence B)
  - **Contradiction** (Sentence A contradicts Sentence B)
  - **Neutral** (No relation)
- **Input Format**:
  `[CLS] Premise sentence [SEP] Hypothesis sentence [SEP]`
- **Segment Embeddings (`token_type_ids`)**:
  - Premise ke tokens $\rightarrow$ `token_type_id = 0`
  - Hypothesis ke tokens $\rightarrow$ `token_type_id = 1`
- BERT ka bidirectional cross-attention dono sentences ke har word ko aapas me compare karta hai, aur `[CLS]` token se final relation classify hota hai.

---

## 8. Common Errors aur Unke Solutions

### Error 1: `RuntimeError: NCCL Error 2: unhandled system error`
- **Kyun hota hai?** Multi-GPU machine par jab aap standard Jupyter notebook me training run karte ho, to Hugging Face `Trainer` PyTorch ke legacy `torch.nn.DataParallel` ko use karta hai. Shared/virtualized servers par direct P2P NVLink access restrict hone par NCCL broadcast crash kar jata hai.
- **Fix**: Notebook ke sabse pehle cell me single GPU bind kar do:
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  os.environ["NCCL_DEBUG"] = "INFO"
  os.environ["NCCL_P2P_DISABLE"] = "1"
  os.environ["NCCL_IB_DISABLE"] = "1"
  ```

### Error 2: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'logging_dir'`
- **Kyun hota hai?** `transformers >= 4.40` aur v5.x me `logging_dir` parameter remove ho chuka hai.
- **Fix**: `logging_dir` ko hata do. Hugging Face automatically `output_dir` ke andar logs store karta hai.

### Error 3: `403 Forbidden: You don't have the rights to create a model under the namespace "..."`
- **Kyun hota hai?** Hugging Face Hub par repository ka naam `"{username}/{repo_name}"` hona chahiye. Agar aap bina username ke koi arbitrary folder name doge, to Hugging Face use organization samajh lega aur 403 error dega.
- **Fix**:
  ```python
  from huggingface_hub import HfApi
  username = HfApi().whoami()["name"]  # e.g., 'Akhand108'
  repo_id = f"{username}/my-bert-imdb"
  ```

---

## 9. Quick Revision Cheat Sheet

| Task | Hugging Face Model Class | Input Format | Labels / Target | Standard LR | Standard Epochs |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Sentiment / Topic** | `BertForSequenceClassification` | `[CLS] Text [SEP]` | Single class ID (`0` to `C-1`) | `2e-5` to `3e-5` | 2 - 3 |
| **Token Tagging / NER**| `BertForTokenClassification` | `[CLS] Words... [SEP]` | Label vector (subwords = `-100`) | `3e-5` to `5e-5` | 3 - 5 |
| **Question Answering** | `BertForQuestionAnswering` | `[CLS] Q [SEP] Context [SEP]` | Span `[start_idx, end_idx]` | `3e-5` | 2 - 3 |
| **Pair Classification**| `BertForSequenceClassification` | `[CLS] A [SEP] B [SEP]` | Relation ID (`token_type_ids`) | `2e-5` | 3 |

### 5 Golden Rules of Fine-Tuning BERT:
1. **Low Learning Rate**: Hamesha `2e-5` se `5e-5` ke beech rahein.
2. **Warmup Use Karein**: Shuruati 10% steps me learning rate warmup zaroor dein.
3. **Clip Gradients**: `max_norm=1.0` se gradient explosion se bachein.
4. **Subword Masking**: Token classification me second subwords ko `-100` dein.
5. **Length Optimization**: Agar texts chote hain to `max_length=256` use karein taaki $4\times$ speedup mile.
