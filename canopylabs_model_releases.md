# Canopy Labs

_Source: <http://canopylabs.ai/model-releases>_

## Introducing Orpheus Speech

[![Image 1: GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/canopyai/Orpheus-TTS)
[![Image 2: Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/canopylabs)
[![Image 3: Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing)

To date, open‑source TTS models have not been competitive with closed‑source models [[1]](https://huggingface.co/spaces/TTS-AGI/TTS-Arena). Nor have TTS models been capable of expressing empathy, consistent with the emotional intelligence of a human.

> _Your browser does not support the video tag._

We’re introducing **Orpheus**, a family of state‑of‑the‑art speech‑LLMs for human‑level speech generation. We're releasing a pretrained and finetuned model in four sizes based on the Llama architecture:

* **Medium** – 3 B parameters
* **Small** – 1 B parameters
* **Tiny** – 400 M parameters
* **Nano** – 150 M parameters

We demonstrate extremely high‑quality, aesthetically pleasing speech generation even with very tiny model sizes.

Our finetuned models, trained on a selection of voices, can be used in production. We also offer our base models along with sample finetuning scripts which can be used for zero‑shot voice cloning and your own finetuning.

We also offer code to do realtime streaming in a very simple Python package. Streaming inference is faster than playback even on an A100 40GB for the 3 B‑parameter model.  
[(see our Google Colab notebook)](https://colab.research.google.com/drive/1xxPpBwI4l_nKUx0J0nzZTtikfqP3UJ6p?usp=sharing)

---

## Try a Demo

We have set up easy inference for both the pretrained and finetuned models. Check out the links below to see the models in action!

---

## Technical Overview

![Architecture](https://canopylabs.ai/assets/images/architecture.png)

### Architecture of Model

Our pretrained model uses **Llama‑3b** as the backbone. We trained it on over **100 k hours** of English speech data and **billions of text tokens**. Training on text tokens boosts its performance on TTS tasks as it maintains a great understanding of language. Below we explore some interesting emergent capabilities of the model.

We use the exact same architecture and training method to train end‑to‑end speech models and plan to release an open‑source end‑to‑end speech model in the coming weeks.

### Handling Disfluencies

| Orpheus (Ours) | ElevenLabs | PlayHT |
|---|---|---|
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |

### Natural Zero‑Shot Voice Cloning (Pretrained Model)

While our pretrained model has not been trained on any voice‑cloning objective, zero‑shot voice cloning can emerge due to the large amounts of pretraining data. Our model chooses natural intonation and emotion, at or exceeding the level of leading models.

**Voice of Prompt**

Our model has not seen this voice during training. The voice is passed to the prompt, which is the first time the model is exposed to it.

| Orpheus | ElevenLabs | PlayHT |
|---|---|---|
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |
| _audio_ | _audio_ | _audio_ |

> Orpheus: Sample was passed in prompt along with text to generate  
> ElevenLabs & PlayHT: Sample was given to instant voice cloning

### Guided Emotion and Intonation

We can teach the base model to speak with a specific emotion using a few dozen high‑quality finetuning examples (text‑speech pairs plus emotion tags).

| Audio | Prompt |
|---|---|
| _audio_ | He qualified for the national tournament. `<normal>` |
| _audio_ | He qualified for the national tournament. `<slow>` |
| _audio_ | He qualified for the national tournament. `<crying>` |
| _audio_ | He qualified for the national tournament. `<sleepy>` |
| _audio_ | `<sigh/>` He qualified for the national tournament. `<normal>` |
| _audio_ | The, uhm, men at those, `<chuckle>`, fundraisers are always **SO** serious. `<normal>` |

---

## In‑Production Usage

Our models are highly accurate, expressive, and customizable thanks to their LLM architecture. The large support for Llama models in the ecosystem and the vast amounts of audio and text data we have extend the models' capabilities.

### Realtime Usage

Realtime usage enables conversational use cases. Our model supports realtime output streaming with very low latency (~200 ms). For even lower latency, **input streaming** of text into the KV cache can reduce latencies to ~25‑50 ms.

### Model Design

We chose two design paradigms that go against convention for realtime speech‑LLMs.

![Tokenizer Architecture](https://canopylabs.ai/assets/images/tokeniser.png)

_SNAC samples tokens at different frequencies which we flatten as shown_

We get **7 tokens per frame** which we decode as a single flattened sequence rather than using seven LM heads. This increases the number of steps the model is required to generate, but the model still generates tokens faster than realtime playback using a straightforward vLLM implementation on an A100 or H100 GPU.

We use a **non‑streaming (CNN‑based) tokenizer**. Other speech LLMs that use SNAC as the decoder suffer from popping between frames fed into the detokenizer. We offer a simple sliding‑window modification to the detokenizer implementation which enables streaming with no popping.

---

_~ The Canopy Labs Team_
