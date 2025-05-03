# Orpheus TTS (中文版)

#### 更新 🔥
- [4/2025] 我们以研究预览版的形式发布了一个[多语言模型系列](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)。
- [4/2025] 我们发布了一份[训练指南](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training)，解释了我们如何创建这些模型，希望能够在此基础上创建出已发布语言和新语言的更好版本。
- 我们欢迎在此[讨论区](https://github.com/canopyai/Orpheus-TTS/discussions/123)提出反馈、批评和问题。

## 概述
Orpheus TTS 是一个基于 Llama-3b 主干的最先进（SOTA, State-of-the-Art）的开源文本转语音系统。Orpheus 展示了使用大型语言模型（LLM, Large Language Models）进行语音合成的涌现能力。

[查看我们的原始博客文章](https://canopylabs.ai/model-releases)

https://github.com/user-attachments/assets/ce17dd3a-f866-4e67-86e4-0025e6e87b8a

https://canopylabs.ai/releases/orpheus_can_speak_any_language 这篇博客整体介绍了如何支持多语言，以及对应多语言的voice和Supported Tags

| 语言 (Language) | 声音 (Voices)          | 支持的标签 (Supported Tags)                                        |
|-----------------|--------------------------|------------------------------------------------------------------|
| French          | pierre, amelie, marie    | chuckle, cough, gasp, groan, laugh, sigh, sniffle, whimper, yawn |
| German          | jana, thomas, max        | chuckle, cough, gasp, groan, laugh, sigh, sniffle, yawn            |
| Korean          | 유나, 준서             | 한숨, 헐, 헛기침, 훌쩍, 하품, 낄낄, 신음, 작은 웃음, 기침, 으르렁        |
| Hindi           | ऋतिका (more coming)     | coming soon                                                      |
| Mandarin        | 长乐, 白芷             | 嬉笑, 轻笑, 呻吟, 大笑, 咳嗽, 抽鼻子, 咳                            |
| Spanish         | javi, sergio, maria    | groan, chuckle, gasp, resoplido, laugh, yawn, cough              |
| Italian         | pietro, giulia, carlo    | sigh, laugh, cough, sniffle, groan, yawn, gemito, gasp           |

## 能力

- **类人语音**: 自然的语调、情感和节奏，优于最先进的闭源模型
- **零样本语音克隆 (Zero-Shot Voice Cloning)**: 无需事先微调即可克隆声音
- **引导式情感和语调**: 使用简单的标签控制语音和情感特征
- **低延迟**: 实时应用约 200 毫秒的流式延迟，通过输入流可降至约 100 毫秒

## 模型

我们提供 2 个英文模型，此外我们还提供数据处理脚本和样本数据集，以便用户可以非常直接地创建自己的微调模型。

1. [**微调生产版 (Finetuned Prod)**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – 适用于日常 TTS 应用的微调模型
2. [**预训练版 (Pretrained)**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – 我们基于 10 万多小时英语语音数据训练的基础模型

我们还在研究发布中提供了一系列多语言模型。

1. [**多语言系列 (Multilingual Family)**](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) - 7 对预训练和微调模型。

### 推理 (Inference)

#### Colab 上的简单设置

我们为各种语言提供了标准化的提示格式，这些 notebook 展示了如何在英语中使用我们的模型。

1. [微调模型 Colab](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (非流式，实时流式请参见下文) – 适用于日常 TTS 应用的微调模型。
2. [预训练模型 Colab](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – 此 notebook 用于条件生成，但可以扩展到一系列任务。

#### 流式推理示例

1. 克隆此仓库
   ```powershell
   git clone https://github.com/canopyai/Orpheus-TTS.git
   ```
2. 导航并安装包
   ```powershell
   cd Orpheus-TTS
   # 安装gcc编译器（若没有安装）
   sudo apt update                # 更新软件源
   sudo apt install build-essential

   # 从源码编译
   cd orpheus_tts_pypi
   pip install . # 内部使用 vllm 实现快速推理，安装vllm会自动安装cuda 
   ```
   vllm 在 3 月 18 日推送了一个稍有问题的版本，因此可以通过在 `pip install orpheus-speech` 之后执行 `pip install vllm==0.7.3` 来还原以解决一些错误。
4. 运行以下示例：test.py(中文语音示例)

#### 附加功能

1. 为您的音频添加水印：使用 Silent Cipher 为您的音频生成添加水印；请参阅[水印音频实现](additional_inference_options/watermark_audio)了解实现方法。

2. 对于使用 Llama cpp 进行无 GPU 推理，请参阅实现[文档](additional_inference_options/no_gpu/README.md)获取实现示例。

#### 提示 (Prompting)

1. `finetune-prod` 模型：对于主模型，您的文本提示格式为 `{name}: 我去了...`。英文版的 `name` 选项按对话真实感（主观基准）排序为 "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe" - 每种语言都有不同的声音 [在此查看声音](https://canopylabs.ai/releases/orpheus_can_speak_any_language#info))。我们的 Python 包会为您完成此格式化，notebook 也会预先添加适当的字符串。您还可以添加以下情感标签：`<laugh>` (笑), `<chuckle>` (轻笑), `<sigh>` (叹气), `<cough>` (咳嗽), `<sniffle>` (吸鼻子), `<groan>` (呻吟), `<yawn>` (打哈欠), `<gasp>` (喘气)。对于多语言模型，请参阅此[文章](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)了解支持的标签。

2. 预训练模型：您可以仅基于文本生成语音，也可以在提示中基于一个或多个现有的文本-语音对来生成语音。由于该模型尚未明确针对零样本语音克隆目标进行训练，因此您在提示中传递的文本-语音对越多，它以正确语音生成的可靠性就越高。


此外，像常规 LLM 一样使用 `temperature`、`top_p` 等常规 LLM 生成参数。为了稳定生成，需要设置 `repetition_penalty>=1.1`。增加 `repetition_penalty` 和 `temperature` 会使模型语速加快。


## 微调模型 (Finetune Model)

以下是如何在任何文本和语音上微调模型的概述。
这是一个非常简单的过程，类似于使用 Trainer 和 Transformers 微调 LLM。

大约 50 个示例后您应该开始看到高质量的结果，但为了获得最佳效果，目标是每个说话者 300 个示例。

1. 您的数据集应该是 [此格式](https://huggingface.co/datasets/canopylabs/zac-sample-dataset) 的 Hugging Face 数据集。
2. 我们使用[此 notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing) 准备数据。这将一个中间数据集推送到您的 Hugging Face 帐户，您可以将其提供给 `finetune/train.py` 中的训练脚本。预处理每千行应花费不到 1 分钟。
3. 修改 `finetune/config.yaml` 文件以包含您的数据集和训练属性，然后运行训练脚本。您还可以运行任何类型的 Hugging Face 兼容过程（如 LoRA）来调整模型。
   ```powershell
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login # <输入您的 HF token>
    wandb login # <输入您的 wandb token>
    accelerate launch train.py
   ```
### 附加资源
1. [使用 unsloth 进行微调](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)
   
## 预训练模型 (Pretrain Model)

这是一个非常简单的过程，类似于使用 Trainer 和 Transformers 训练 LLM。

提供的基础模型训练超过 10 万小时。我建议不要使用合成数据进行训练，因为当您尝试微调特定声音时，它会产生较差的结果，可能是因为合成声音缺乏多样性，并且在分词时映射到相同的标记集（即导致码本利用率低下）。

我们在长度为 8192 的序列上训练 3b 模型 - 对于 `<TTS-dataset>` 预训练，我们使用与 TTS 微调相同的数​​据集格式。我们将 `input_ids` 序列连接在一起以提高训练效率。所需的文本数据集格式在此 issue [#37](https://github.com/canopyai/Orpheus-TTS/issues/37) 中描述。

如果您要对此模型进行扩展训练，例如针对另一种语言或风格，我们建议从仅微调开始（无文本数据集）。文本数据集背后的主要思想在博客文章中讨论过。（长话短说：它不会忘记太多的语义/推理能力，因此能够更好地理解如何为短语赋予语调/表达情感，然而，大部分遗忘会发生在训练的早期，即 <100000 行），因此除非您进行非常扩展的微调，否则可能不会有太大区别。

## 另请查看

虽然我们无法验证这些实现的完全准确性/无错误性，但它们已在一些论坛上被推荐，因此我们在此列出：

1. [使用 LM Studio API 在本地运行 Orpheus TTS 的轻量级客户端](https://github.com/isaiahbjork/orpheus-tts-local)
2. [OpenAI 兼容的 Fast-API 实现](https://github.com/Lex-au/Orpheus-FastAPI)
3. [由 MohamedRashad 好心设置的 HuggingFace Space](https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS)
4. [可在 WSL 和 CUDA 上流畅运行的 Gradio WebUI](https://github.com/Saganaki22/OrpheusTTS-WebUI)


# 清单 (Checklist)

- [x] 发布 3b 预训练模型和微调模型
- [ ] 发布参数量为 1b、400m、150m 的预训练和微调模型
- [ ] 修复实时流式包中偶尔跳帧的小故障。
- [ ] 修复语音克隆 Colab notebook 实现。 