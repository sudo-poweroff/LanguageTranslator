# LanguageTranslator
  This repository provides a **multilingual machine translation system** that converts input text from various European languages into English. By leveraging the power of **Transformer architectures** and language-specific models, the system delivers efficient translation with a user-friendly interface.

## 🧠 Domain Overview
  The project addresses the task of automated machine translation. Specifically, it translates sentences from multiple source languages to English using deep learning models. The system includes:
  - Custom-trained **Transformer models** for each language pair.
  - A **Language Identifier** that detects the input language (from _langid_ library).
  - A **routing-based multilingual translation pipeline** that dynamically selects the appropriate model.
  - A **Gradio interface** for seamless interaction.

This modular approach ensures high flexibility, retrainability, and improved compatibility with low-resource computing environments.

## 🗂️ Dataset
  The training corpora were extracted from the _opus-books_ dataset available on Hugging Face. We selected key European language pairs:
  | Language Pair | # Sentences |
  | :---:         | :---:       |
  | en-fr         | 127085      |
  | en-ru         | 17496       |
  | en-it         | 32332       |
  | en-pl         | 2831        |
  | en-fi         | 3645        |
  | en-nl         | 38700       |
  | en-pt         | 1404        |
  | de-en         | 51500       |
  
  Each dataset was split as follows:
  - **Train**: 70%
  - **Validation**: 10%
  - **Test**: 20%

## 🤖 Model Architecture

  ### Why Multiple Models?
  Due to:
  - Significant variation in sentence lengths across languages
  - Memory and computational constraints
  - Desire to avoid excessive padding and resource usage

  We decided to train **one Transformer model per language pair**. This enables:
  - Specialized training for each pair
  - Parallel training on different machines
  - Easy model retraining without affecting others

  Each model is part of a larger **routing system**, with a **language identifier** determining the correct model at runtime.

  ### Transformer Details
  Each model is based on the standard Transformer architecture and includes:
  - Embedding Layer with positional encodings
  - 6 Encoder + 6 Decoder blocks
  - Multi-head Attention, Residual Connections, Feedforward Layers
  - Projection Layer for output token prediction

  ⚙️ Hyperparameters:
  - **Embedding Dimension**: Varies by model
  - **Batch Size**: 8
  - **Epochs**: 20
  - **Learning Rate**: 1e-4

  ### 🦾 Pre-trained models
  To evaluate our models against the current state-of-the-art, we selected one pretrained model for each language pair, all sourced from Hugging Face. The table below lists the models used:
  | Language pair | Pre-trained model | Trained on |
  | :---:         | :---:             | :---:      |
  | fr-en         | [Helsinki-NLP/opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) | opus |
  | ru-en         | [Helsinki-NLP/opus-mt-ru-en](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en) | opus |
  | it-en         | [Helsinki-NLP/opus-mt-it-en](https://huggingface.co/Helsinki-NLP/opus-mt-it-en) | opus |
  | pl-en         | [Helsinki-NLP/opus-mt-pl-en](https://huggingface.co/Helsinki-NLP/opus-mt-pl-en) | opus |
  | fi-en         | [Helsinki-NLP/opus-tatoeba-fi-en](https://huggingface.co/Helsinki-NLP/opus-tatoeba-fi-en) | opus |
  | nl-en         | [Helsinki-NLP/opus-mt-nl-en](https://huggingface.co/Helsinki-NLP/opus-mt-nl-en) | opus |
  | pt-en         | [unicamp-dl/translation-pt-en-t5](https://huggingface.co/unicamp-dl/translation-pt-en-t5) | 6 different datasets |
  | de-en         | [Helsinki-NLP/opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) | opus |

## 💻 Project environment setup
  To replicate this project, follow this steps:
  1. Clone the repository by running:
  ```bash
  git clone https://github.com/sudo-poweroff/LanguageTranslator.git
  ```
  2. Make sure to install the required dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

## 📊 Results
We evaluated all models using the following metrics:
- **CER** (Character Error Rate)
- **WER** (Word Error Rate)
- **BLEU** (Bilingual Evaluation Understudy)
- **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)

### 🧪 Custom-Trained Models
| Dataset | CER | WER | BLEU | Meteor |
| :---:   | :---: | :---: | :---: | :---: |
| en-fr   | 0.6758 | 1.1093 | 0.0768 | 0.4779 |
| en-ru   | 0.6965 | 1.0614 | 0.0331 | 0.3157 |
| en-it   | 0.7334 | 1.12   | 0.0262 | 0.2984 |
| en-pl   | 0.8375 | 1.2141 | 0.0038 | 0.1757 |
| en-fi   | 0.7317 | 1.0708 | 0.0097 | 0.2483 |
| en-nl   | 0.7322 | 1.1275 | 0.0342 | 0.359  |
| en-pt   | 0.6992 | 1.0925 | 0      | 0.2941 |
| de-en   | 0.7051 | 1.0892 | 0.0403 | 0.3633 |

### 🦾 Pre-trained models
| Dataset | CER | WER | BLEU | Meteor |
| :---:   | :---: | :---: | :---: | :---: |
| en-fr   | 0.5011 | 0.6947 | 0.1869 | 0.526 | 1 |
| en-ru   | 0.5226 | 0.7123 | 0.1484 | 0.4877 | 0 |
| en-it   | 0.5757 | 0.7757 | 0.1267 | 0.4421 | 0 |
| en-pl   | 0.7081 | 0.898  | 0.0572 | 0.3096 | 0 |
| en-fi   | 0.6038 | 0.8084 | 0.1201 | 0.4171 | 0 |
| en-nl   | 0.6339 | 0.8407 | 0.1334 | 0.4691 | 0 |
| en-pt   | 0.7131 | 0.8235 | 0.07   | 0.3245 | 0 |
| de-en   | 0.5731 | 0.7697 | 0.151  | 0.4987 | 1 | 

### 📌 Results Overview
Our **custom-trained models** obtained **high CER and WER scores**, reflecting a relatively poor translation performance. This is primarily attributed to the **limited size and diversity of the training datasets**.

On the other hand, the **pre-trained models** consistently outperformed the custom models, benefiting from more extensive training data and optimized architectures. However, even their performance metrics fall short of optimal standards, especially on longer or more complex inputs.

## 🛠️ Future works
- [ ] Support many-to-many translation, not just source-to-English.
- [ ] Use larger and multilingual corpora with translation alternatives.

## 👨🏻‍💻 Contributors
<a href="https://github.com/sudo-poweroff/LanguageTranslator/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sudo-poweroff/LanguageTranslator" />
</a>
