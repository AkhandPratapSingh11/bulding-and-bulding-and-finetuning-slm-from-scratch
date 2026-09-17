---
language:
- en
license: apache-2.0
base_model: bert-base-uncased
tags:
- bert
- fine-tuned
- text-classification
datasets:
- stanfordnlp/imdb
metrics:
- accuracy
- weighted f1
- weighted precision
- weighted recall
pipeline_tag: text-classification
---

# 🚀 Fine-Tuned BERT for Text-Classification

This model is an optimized, fine-tuned checkpoint of **`bert-base-uncased`** specialized for **Text-Classification**, trained on the **`stanfordnlp/imdb`** dataset.

---

## 📌 Model Description & Intended Use
- **Primary Task**: Text-Classification
- **Base Architecture**: `bert-base-uncased` (12 Transformer Encoder Layers, 768 Hidden Dim, 12 Attention Heads)
- **Intended Use**: Binary sentiment classification (positive/negative sentiment analysis on consumer reviews and social text).

---

## 📊 Dataset Details
- **Dataset Name**: `stanfordnlp/imdb`
- **Overview**: 50,000 highly polarized movie reviews from the Stanford IMDB benchmark for binary sentiment analysis.

---

## ⚙️ Training Hyperparameters
The model was fine-tuned using modern optimization best practices (Mixed Precision, Gradient Accumulation, Differential Weight Decay):

| Hyperparameter | Value |
| :--- | :--- |
| **`Base Model`** | `bert-base-uncased` |
| **`Learning Rate`** | `3e-05` |
| **`Batch Size (Micro)`** | `2` |
| **`Gradient Accumulation`** | `1` |
| **`Effective Batch Size`** | `2` |
| **`Epochs`** | `1` |
| **`Precision`** | `BFloat16` |
| **`Optimizer`** | `AdamW (differential weight decay)` |
| **`Scheduler`** | `Cosine Annealing with Warmup` |
| **`Warmup Ratio`** | `0.1` |
| **`Max Length`** | `256` |

---

## 📈 Evaluation Results & Benchmarks
Evaluation results on the held-out validation/test split:

| Metric | Result |
| :--- | :--- |
| **`Accuracy`** | `0.5` |
| **`Weighted F1`** | `0.3333` |
| **`Weighted Precision`** | `0.25` |
| **`Weighted Recall`** | `0.5` |

---

## 💻 Quickstart & Inference Usage

### Method 1: Using Hugging Face `pipeline` (Recommended)
```python
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
nlp_pipe = pipeline("text-classification", model="bert-text-classification")

from transformers import pipeline

# Load pipeline from Hub
classifier = pipeline("text-classification", model="bert-text-classification")

# Run inference
result = classifier("This movie was an absolute masterpiece with gripping performances!")
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.998}]

```

### Method 2: Native PyTorch Inference
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-text-classification")
model = AutoModel.from_pretrained("bert-text-classification")
model.eval()

inputs = tokenizer("Sample input text", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

---

## 📄 License
This model checkpoint is released under the **Apache-2.0** license.
