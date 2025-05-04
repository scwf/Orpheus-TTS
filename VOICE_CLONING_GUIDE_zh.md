# Orpheus TTS 声音克隆指南

本文档详细介绍了使用 Orpheus TTS 克隆你自己声音的两种主要方法：**微调模型** 和 **零样本语音克隆**。

## 方法一：微调模型 (推荐)

这是获得高质量、稳定且个性化的声音克隆效果的最佳方法。此过程涉及使用你自己的语音数据对预训练的 Orpheus TTS 模型进行微调。

**优点:**
*   生成的声音质量高，更接近你的原始声音。
*   效果更稳定可靠。
*   可以更好地捕捉你独特的说话风格和韵律。

**缺点:**
*   需要准备一定量的语音数据（推荐 300 条左右）。
*   需要进行模型训练，这需要时间和计算资源（GPU）。

**详细步骤:**

1.  **准备你自己的语音数据集**
    *   **录制语音**: 录制一批包含你自己声音的音频片段。确保录音环境安静，音频清晰。
    *   **准备文本**: 为每个音频片段准备对应的文本转录。
    *   **整理格式**: 将音频文件和对应的文本整理成 Hugging Face 数据集格式。你需要创建一个包含 `audio` 和 `text` 列的数据集。
        *   `audio` 列应包含音频文件的路径或音频数据本身 (推荐使用 Hugging Face 的 `Audio` 特征)。确保音频采样率为 24kHz。
        *   `text` 列应包含对应的文本转录。
    *   **参考示例**: 你可以参考官方提供的示例数据集结构：[`canopylabs/zac-sample-dataset`](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
    *   **数据量建议**: 
        *   至少需要约 50 个样本才能看到初步效果。
        *   为了获得最佳效果，建议为每个说话人（即你自己）准备大约 **300 个样本**。

2.  **数据预处理**
    *   **使用官方 Notebook**: Orpheus TTS 提供了一个 Colab notebook 来帮助你预处理数据集，将其转换为模型训练所需的格式。
        *   **Notebook 链接**: [数据准备 Notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing)
    *   **处理过程**: 这个 notebook 会加载你的原始数据集，进行必要的处理（如特征提取），并将处理后的中间数据集上传到你的 Hugging Face Hub 账户。此过程通常很快（每千条数据约 1 分钟）。记下这个上传后的数据集名称（例如 `your-username/your-processed-dataset-name`）。

3.  **配置训练参数**
    *   **找到配置文件**: 在你的 Orpheus-TTS 项目克隆中，找到 `finetune/config.yaml` 文件。
    *   **修改配置**: 打开并编辑此文件，至少需要修改以下部分：
        *   `dataset_name`: 将其值更改为你在**步骤 2** 中上传到 Hugging Face Hub 的 **处理后的** 数据集名称。
        *   `output_dir`: 指定训练完成后模型保存的本地路径或 Hugging Face Hub 仓库名称。
        *   （可选）你还可以根据需要调整其他超参数，如 `learning_rate`, `num_train_epochs`, `per_device_train_batch_size` 等。

4.  **开始训练**
    *   **安装依赖**: 确保你的 Python 环境安装了所有必要的库。在你的项目根目录下的 PowerShell 终端中运行：
        ```powershell
        pip install transformers datasets wandb trl flash_attn torch accelerate
        ```
        *(注意: `flash_attn` 通常需要从源码编译或需要特定 CUDA 版本的预编译包)*
    *   **登录平台**: 为了保存模型和追踪训练过程，你需要登录到 Hugging Face 和 Weights & Biases (wandb)。
        ```powershell
        huggingface-cli login # 按照提示输入你的 Hugging Face API 令牌 (Token)
        wandb login           # 按照提示输入你的 wandb API 密钥 (Key)
        ```
    *   **启动训练**: 导航到 `finetune` 目录，并使用 `accelerate` 启动训练脚本。
        ```powershell
        cd finetune
        accelerate launch train.py
        ```
    *   **训练时长**: 训练所需时间取决于你的数据集大小、GPU 性能和配置的训练周期数。

5.  **使用克隆后的模型**
    *   训练完成后，微调过的模型将保存在你 `config.yaml` 中指定的 `output_dir`。如果指定的是 Hugging Face Hub 仓库名，模型会自动上传。
    *   你可以像加载官方模型一样加载你自己的微调模型进行推理。生成的语音将带有你克隆的声音特征。
        ```python
        from orpheus_tts import OrpheusModel
        
        # 加载你自己的微调模型 (替换成你的模型路径或HF仓库名)
        my_model = OrpheusModel(model_name="path/to/your/finetuned_model_directory_or_hf_repo") 
        
        prompt = "这是用我克隆的声音生成的语音。"
        # 注意：微调后，你通常需要使用特定的说话人标识符，
        # 这可能是在数据准备或训练配置中定义的，或者你可以省略 'voice' 参数
        # syn_tokens = my_model.generate_speech(prompt=prompt, voice="your_voice_id") 
        syn_tokens = my_model.generate_speech(prompt=prompt)
        
        # ... 后续处理和保存音频文件的代码 ...
        ```

---

## 方法二：零样本语音克隆 (使用预训练模型)

这种方法利用 Orpheus TTS 预训练模型 (`canopylabs/orpheus-tts-0.1-pretrained`) 的能力，通过在推理时提供目标声音的样本来尝试模仿该声音，**无需进行额外的模型训练**。

**优点:**
*   简单快捷，无需准备大量数据和进行模型训练。
*   适用于快速尝试或对声音相似度要求不高的场景。

**缺点:**
*   效果不如微调稳定和准确，声音相似度可能有限。
*   对输入样本的质量和数量敏感。

**详细步骤:**

1.  **准备语音样本**: 准备一个或多个你想要克隆的声音的【音频文件】和对应的【文本转录】。
2.  **加载预训练模型**: 加载官方提供的预训练模型。
    ```python
    from orpheus_tts import OrpheusModel
    
    # 加载预训练模型
    pretrained_model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-pretrained")
    ```
3.  **构建提示 (Prompt)**: 在调用 `generate_speech` 时，将你的语音样本（音频文件路径和文本）作为上下文信息传递给模型。具体的实现方式可能依赖于 `orpheus_tts` 库的具体接口设计，但概念上类似：
    *   你需要将【文本-语音】对组合成模型能够理解的格式。查阅 `orpheus_tts` 包关于预训练模型推理的文档或示例以获取准确的格式。
    *   `README` 文件指出，提供的【文本-语音】对越多，效果越好。

4.  **生成语音**: 调用生成函数，提供你想要合成的新文本。
    ```python
    # 概念性示例，具体 API 可能不同
    # 假设 sample_audio_path_1, sample_text_1 等是你的样本
    context_prompt = f"{sample_text_1} [AUDIO] {sample_audio_path_1} {sample_text_2} [AUDIO] {sample_audio_path_2}"
    new_text = "这是尝试零样本克隆我的声音。"
    
    # 具体的调用方式需要参考库的文档
    # syn_tokens = pretrained_model.generate_speech(prompt=new_text, context=context_prompt)
    
    # ... 后续处理和保存音频文件的代码 ... 
    ```
    **注意**: 请务必查阅 Orpheus TTS 关于如何使用预训练模型进行条件生成的最新文档或示例代码，以了解正确的 API 调用方式。

---

## 总结

*   **追求高质量和个性化**: 选择 **方法一：微调模型**。
*   **快速尝试或要求不高**: 可以尝试 **方法二：零样本语音克隆**。

对于大多数严肃的应用场景，**强烈推荐使用微调模型**的方法来获得最佳的声音克隆效果。 