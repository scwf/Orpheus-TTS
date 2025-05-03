# Canopy Labs – Orpheus Multilingual Research Release
_Source: <https://canopylabs.ai/releases/orpheus_can_speak_any_language>_

## Research Release for Fr, De, Es, It, Zh, Ko, Hi and a training manual for **any** language

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/canopyai/Orpheus-TTS)
[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/canopylabs)
[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing)

Existing speech models struggle to speak languages other than English because their architectures do not scale well to low‑resource languages. **Orpheus Multilingual** is a family of state‑of‑the‑art speech‑LLMs for highly expressive speech across many languages.

We have seen dozens of developers create multilingual versions of Orpheus 0.1, so we provide a detailed, inexpensive training guide.

> _Your browser does not support the video tag._

### Pretrained & Finetuned Models

Available today:

* [French](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [German](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [Spanish](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [Italian](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [Chinese (Mandarin)](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [Korean](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)
* [Hindi](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)

Even with limited multilingual data, these models generate aesthetically pleasing speech. They follow the same architecture and prompt format as the original pretrained models.

---

## Speaking Naturally

Models should:

1. Select appropriate emotional tonality.  
2. Produce human non‑speech sounds (laughing, sighing, etc.).  
3. Render disfluencies and self‑interruption realistically.

| French | German | Mandarin |
|---|---|---|
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |

---

## Training Overview

We **pretrain**, then **finetune** on language‑specific data.  
All weights start from **Orpheus‑3b‑0.1‑pretrained**.

### Pretraining Data

| Language(s) | Hours (k) |
|---|---|
| French | 5 |
| German | 5 |
| Mandarin | 20 |
| Korean | 5 |
| Hindi | 1 |
| Spanish | 1 |
| Italian | 1 |

Sequences are concatenated up to length 8192.

![German pretraining loss](https://canopylabs.ai/assets/images/de-pretraining.png)

#### Tips to improve pretraining

* **More data** – 5 k h is a baseline; English uses 100 k h.  
* **Train on text tokens** – As with English, adding language text boosts semantics.

### Finetuning Overview

Professional actors record **300 lines** each (≥2 per language).  
Hyper‑parameters:

| Hyper‑parameter | Value |
|---|---|
| Learning rate | 5 × 10⁻⁵ |
| Machines | 1 |
| Trainable params | All |
| LR schedule | Cosine decay |
| Batch size | 1 |
| Precision | bf16 |

Finetuning curves show that language‑specific pretraining greatly improves convergence.

![German finetune steps](https://canopylabs.ai/assets/images/de-tune-steps.png)

### Tag Frequencies (Korean)

| Tag | Count |
|---|---|
| 한숨 | 93 |
| 헐 | 72 |
| 헛기침 | 58 |
| 훌쩍 | 57 |
| 하품 | 51 |
| 낄낄 | 50 |
| 신음 | 25 |
| 작은 웃음 | 16 |
| 기침 | 14 |
| 으르렁 | 10 |
| 훌쩍임 | 8 |
| 약간 웃음 | 7 |
| 으르렁거림 | 6 |
| 킥킥 | 5 |

---

## Understanding Pretraining Data

* **Synthetic single‑speaker data** can cause catastrophic forgetting.  
* **Sample rate mismatches** hurt quality; prefer native 24 kHz or downsample finetune data.

---

## Model Release Voices

| Language | Voices | Supported tags |
|---|---|---|
| French | pierre, amelie, marie | chuckle, cough, gasp, groan, laugh, sigh, sniffle, whimper, yawn |
| German | jana, thomas, max | chuckle, cough, gasp, groan, laugh, sigh, sniffle, yawn |
| Korean | 유나, 준서 | 한숨, 헐, 헛기침, 훌쩍, 하품, 낄낄, 신음, 작은 웃음, 기침, 으르렁 |
| Hindi | ऋतिका | coming soon |
| Mandarin | 长乐, 白芷 | 嬉笑, 轻笑, 呻吟, 大笑, 咳嗽, 抽鼻子, 咳 |
| Spanish | javi, sergio, maria | groan, chuckle, gasp, resoplido, laugh, yawn, cough |
| Italian | pietro, giulia, carlo | sigh, laugh, cough, sniffle, groan, yawn, gemito, gasp |

### Performance Snapshot

😀 = highest 😐 = medium 😔 = lowest

| Language | Pretrained | Finetuned |
|---|---|---|
| French | 😀 | 😐 |
| German | 😀 | 😐 |
| Korean | 😀 | 😐 |
| Hindi | 😔 | 😔 |
| Mandarin | 😀 | 😐 |
| Spanish | 😔 | 😔 |
| Italian | 😔 | 😔 |

---

## Conclusion

**Orpheus** offers a simpler, higher‑quality, and more customizable pathway to multilingual speech models. We hope these research models, guide, and tuning code empower you to develop your own.

> _We have no immediate plans for further multilingual work; we welcome the community to build upon this foundation._  

_~ The Canopy Labs Team_
