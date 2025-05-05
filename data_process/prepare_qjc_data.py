import os
import datasets
from datasets import Dataset, Audio, DatasetDict
from pathlib import Path
import huggingface_hub
import torch
import torchaudio.transforms as T
from tqdm.auto import tqdm # For progress bars
import sys

# Suppress Hugging Face logging noise if desired
# import logging
# logging.getLogger("datasets").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- Configuration ---
# Set your local data directory path (relative to this script)
data_dir = Path("./qjc_sample_data")
# Target sampling rate for audio
target_sampling_rate = 24000
# Hugging Face Hub configuration
# Modify this to your desired final processed dataset name on the Hub
target_repo_name = "qjc-audio-tokenized" # Name for the FINAL tokenized dataset
# Set num_proc to None to disable multiprocessing, or set to a specific number
# Using multiprocessing significantly speeds up map operations but uses more RAM
num_proc = os.cpu_count() - 1 if os.cpu_count() > 1 else None # Use most cores but leave one free
# SNAC model for audio tokenization
# SNAC 模型（全称：Self-supervised Non-autoregressive Acoustic Codebook）是一种用于将音频信号（如语音波形）编码
# 为离散声学标记（acoustic codes，声码）的神经网络模型。SNAC 模型可以将连续的音频波形（如 .wav 文件）转换为一系列离散的"代码"或"token"
# 它常用于现代文本到语音（TTS, Text-to-Speech）系统的数据预处理和特征提取阶段。
snac_model_name = "hubertsiuzdak/snac_24khz"
# Tokenizer for text
text_tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"
# --- Configuration End ---


# --- Special Token IDs ---
# Based on the notebook and typical Llama-based TTS models
tokeniser_length = 128256 # Check if this is correct for the specific tokenizer? Often vocab_size
# It's safer to get these from the loaded tokenizer if possible, but we'll use the notebook values for now.
# However, the notebook values seem extremely high. Let's re-check.
# The notebook uses offset values like +128266. This implies the audio tokens live *outside* the normal vocab.
# Let's define the offsets and base ID. Base ID 128266 might be vocab_size + some_offset.
# It's crucial these match what the finetuning script expects.
# Let's use the structure from the notebook more directly.
audio_tokens_offset = 128266 # Base offset used in the notebook
snac_levels = 7 # SNAC outputs 7 levels of codes in the notebook processing
snac_codebook_size = 4096 # Typical codebook size, used for offsets

# --- Define Special Tokens conceptually ---
# We will add them later if the tokenizer doesn't have them
# These IDs seem to be relative to `tokeniser_length`, let's assume that for now.
# We might need to add these as special tokens to the tokenizer itself.
# Let's load the tokenizer first to check its vocab size and special tokens.
print(f"Loading text tokenizer: {text_tokenizer_name}...")
try:
    from transformers import AutoTokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
    print("Text tokenizer loaded successfully.")
    tokeniser_length = text_tokenizer.vocab_size # Use actual vocab size
    print(f"Tokenizer vocabulary size: {tokeniser_length}")
except Exception as e:
    print(f"错误: 加载文本分词器失败 {text_tokenizer_name}: {e}")
    exit(1)

# Re-evaluate special token IDs based on actual vocab size
# The notebook adds large offsets (e.g., 128266). This usually means audio tokens
# are *added* as new tokens or live in a separate embedding space.
# Let's stick to the notebook's direct calculation using the offset.
# Ensure the finetuning script uses the *same* calculation.

# These control tokens might need to be added to the tokenizer if not present
# For simplicity, we assume the finetuning script expects these *numeric* IDs.
start_of_text = 128000 # From notebook - is this BOS or a custom token?
end_of_text = 128009   # From notebook - is this EOS or a custom token?
start_of_speech = tokeniser_length + 1 # Placeholder, might need adjustment
end_of_speech = tokeniser_length + 2   # Placeholder
start_of_human = tokeniser_length + 3  # Placeholder
end_of_human = tokeniser_length + 4  # Placeholder
start_of_ai = tokeniser_length + 5     # Placeholder
end_of_ai = tokeniser_length + 6     # Placeholder
# pad_token = tokeniser_length + 7     # Usually tokenizer.pad_token_id

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA 可用，将在 GPU 上运行 SNAC 模型。Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("警告: 未检测到 CUDA。将在 CPU 上运行 SNAC 模型，这会非常慢！")

# --- Load SNAC Model ---
print(f"Loading SNAC model: {snac_model_name}...")
try:
    from snac import SNAC
    snac_model = SNAC.from_pretrained(snac_model_name)
    snac_model = snac_model.to(device)
    snac_model.eval() # Set to evaluation mode
    print("SNAC model loaded successfully.")
except ImportError:
    print("错误: 未找到 'snac' 库。请运行 'pip install snac'")
    exit(1)
except Exception as e:
    print(f"错误: 加载 SNAC 模型失败 {snac_model_name}: {e}")
    exit(1)


# --- Helper Functions (Adapted from Notebook) ---

def tokenise_audio(waveform_np, orig_sr):
    """Converts a NumPy audio waveform to SNAC codes."""
    if waveform_np is None or waveform_np.size == 0:
        return None
    try:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(dtype=torch.float32)

        # Resample if necessary
        if orig_sr != target_sampling_rate:
            # print(f"Resampling from {orig_sr} to {target_sampling_rate}") # Debug
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sampling_rate)
            waveform = resampler(waveform)

        waveform = waveform.unsqueeze(0).to(device) # Add batch dim and move to device

        # Generate codes using SNAC model
        with torch.inference_mode():
            # Encode returns list of tensors [level][batch][time]
            codes_tensor_list = snac_model.encode(waveform)
            # codes format from notebook: [L][B=1][T] -> we need [L][T]
            codes = [level[0] for level in codes_tensor_list] # Remove batch dim

        # Process codes using the specific logic from the notebook
        # codes[0]: [T], codes[1]: [2T], codes[2]: [4T]
        all_codes = []
        num_frames_level0 = codes[0].shape[0]
        for i in range(num_frames_level0):
            # Ensure indices are within bounds
            idx1 = 2 * i
            idx1_next = idx1 + 1
            idx2 = 4 * i
            idx2_next1 = idx2 + 1
            idx2_next2 = idx2 + 2
            idx2_next3 = idx2 + 3

            if (idx1_next < codes[1].shape[0] and
                idx2_next3 < codes[2].shape[0]):
                # Append codes with offsets based on level and position within the frame
                all_codes.append(codes[0][i].item() + audio_tokens_offset)                     # Level 0, Pos 0
                all_codes.append(codes[1][idx1].item() + audio_tokens_offset + snac_codebook_size) # Level 1, Pos 0
                all_codes.append(codes[2][idx2].item() + audio_tokens_offset + 2 * snac_codebook_size) # Level 2, Pos 0
                all_codes.append(codes[2][idx2_next1].item() + audio_tokens_offset + 3 * snac_codebook_size) # Level 2, Pos 1
                all_codes.append(codes[1][idx1_next].item() + audio_tokens_offset + 4 * snac_codebook_size) # Level 1, Pos 1
                all_codes.append(codes[2][idx2_next2].item() + audio_tokens_offset + 5 * snac_codebook_size) # Level 2, Pos 2
                all_codes.append(codes[2][idx2_next3].item() + audio_tokens_offset + 6 * snac_codebook_size) # Level 2, Pos 3
            # else: # Handle potential boundary cases if needed, maybe skip last frame?
            #     print(f"Skipping boundary frame {i}")

        if not all_codes: # Handle cases where audio is too short?
             print(f"警告: 音频处理后未生成任何 codes。音频可能太短。")
             return None

        return all_codes

    except Exception as e:
        print(f"错误: tokenise_audio 处理失败: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

def add_codes_map(batch):
    """Map function to apply tokenise_audio to a batch."""
    results = {"codes_list": []}
    # Process audio column if it exists and contains necessary keys
    if "audio" in batch and isinstance(batch["audio"], list):
        for audio_item in batch["audio"]:
             codes = None
             if isinstance(audio_item, dict) and "array" in audio_item and "sampling_rate" in audio_item:
                  codes = tokenise_audio(audio_item["array"], audio_item["sampling_rate"])
             results["codes_list"].append(codes)
    return results


def remove_duplicate_frames(example):
    """Removes consecutive duplicate frames based on the first code of the frame."""
    vals = example.get("codes_list")
    if not vals or len(vals) % snac_levels != 0:
        # Keep as is if invalid or empty
        # print(f"警告: remove_duplicate_frames 的输入无效或长度 ({len(vals) if vals else 0}) 不能被 {snac_levels} 整除。")
        return example # Return original example

    result = vals[:snac_levels] # Keep the first frame
    # removed_frames = 0

    for i in range(snac_levels, len(vals), snac_levels):
        current_first_code = vals[i]
        previous_first_code = result[-snac_levels]

        if current_first_code != previous_first_code:
            result.extend(vals[i:i+snac_levels])
        # else:
        #     removed_frames += 1

    example["codes_list"] = result
    return example

def create_input_ids_map(batch):
    """Map function to create final input_ids, labels, and attention_mask."""
    # Ensure 'text' and 'codes_list' are lists in the batch
    texts = batch.get("text", [])
    codes_lists = batch.get("codes_list", [])
    batch_size = len(texts)

    new_batch = {"input_ids": [], "labels": [], "attention_mask": []}

    for i in range(batch_size):
        text = texts[i]
        codes_list = codes_lists[i]

        # Skip if codes_list is None or empty (error during audio processing)
        if codes_list is None or not codes_list:
             # Append empty lists or handle appropriately
             new_batch["input_ids"].append([])
             new_batch["labels"].append([])
             new_batch["attention_mask"].append([])
             print("警告: 检测到空的 codes_list，跳过此样本的 input_id 创建。")
             continue

        # Tokenize text - use add_special_tokens=False if start/end tokens are manually added
        # The notebook used add_special_tokens=True, let's stick to that for now
        text_ids = text_tokenizer.encode(text, add_special_tokens=True)
        # The notebook added an explicit end_of_text token ID
        text_ids.append(end_of_text)

        # Construct final sequence based on notebook structure
        input_ids = (
            [start_of_human]
            + text_ids
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + codes_list # Already processed codes
            + [end_of_speech]
            + [end_of_ai]
        )
        # Labels are typically the same as input_ids for auto-regressive models
        labels = input_ids[:]
        attention_mask = [1] * len(input_ids)

        new_batch["input_ids"].append(input_ids)
        new_batch["labels"].append(labels)
        new_batch["attention_mask"].append(attention_mask)

    return new_batch


# --- Main Script Logic ---

def main():
    if not data_dir.is_dir():
        print(f"错误: 数据目录不存在或不是一个目录: {data_dir.resolve()}")
        return

    print(f"开始处理目录: {data_dir.resolve()}")

    # 1. Scan directory and create initial Dataset
    print("\n步骤 1: 扫描文件并创建初始数据集...")
    data_list = []
    found_files = False
    for wav_path in tqdm(list(data_dir.glob("*.wav")), desc="Scanning files"):
        found_files = True
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                text_content = txt_path.read_text(encoding='utf-8').strip()
                if text_content:
                    data_list.append({
                        "audio_path": str(wav_path.resolve()), # Store path first
                        "text": text_content
                    })
                else: print(f"  警告: 跳过空文本文件: {txt_path.name}")
            except Exception as e: print(f"  错误: 读取文本文件失败 {txt_path.name}: {e}")
        else: print(f"  警告: 跳过无匹配文本文件的音频: {wav_path.name}")

    if not found_files: print("错误: 未找到 .wav 文件。"); return
    if not data_list: print("错误: 未匹配到任何 audio/text 对。"); return
    print(f"成功匹配 {len(data_list)} 个 audio/text 对。")

    raw_ds = Dataset.from_list(data_list)
    print("初始数据集创建完成。")
    print(raw_ds)

    # 2. Load and process audio using Audio feature (handles loading & resampling)
    print(f"\n步骤 2: 加载音频并重采样至 {target_sampling_rate} Hz...")
    try:
        # Use cast_column to apply the Audio feature loading
        # This replaces the 'audio_path' string column with an 'audio' dict column
        ds_with_audio = raw_ds.cast_column("audio_path", Audio(sampling_rate=target_sampling_rate))
        # Rename the column to 'audio' for consistency
        ds_with_audio = ds_with_audio.rename_column("audio_path", "audio")
        print("音频加载和重采样完成。")
        print("验证第一个样本的音频数据:")
        print(ds_with_audio[0]['audio'])

    except Exception as e:
        print(f"错误: 加载或转换音频时出错: {e}")
        print("请确保所有 .wav 文件都是有效的，并且库（如 soundfile, librosa）已安装。")
        return

    # 3. Tokenize Audio using SNAC
    print("\n步骤 3: 使用 SNAC 模型进行音频分词...")
    # Use batched=True for potential speedup, adjust batch_size if needed
    processed_ds = ds_with_audio.map(
        add_codes_map,
        batched=True,
        batch_size=16, # Adjust based on GPU memory
        remove_columns=["audio"], # Remove raw audio after getting codes
        num_proc=num_proc # Enable multiprocessing if configured
    )
    print("音频分词完成。")

    # 4. Filter out errors (where codes_list is None or empty)
    print("\n步骤 4: 过滤处理失败的样本...")
    original_count = len(processed_ds)
    processed_ds = processed_ds.filter(
        lambda example: example["codes_list"] is not None and len(example["codes_list"]) > 0,
        num_proc=num_proc
        )
    filtered_count = len(processed_ds)
    print(f"过滤完成。移除了 {original_count - filtered_count} 个处理失败的样本。")
    if filtered_count == 0:
        print("错误: 过滤后没有剩余的有效样本。请检查音频文件和处理日志。")
        return

    # 5. Remove duplicate frames
    print("\n步骤 5: 移除连续的重复音频帧...")
    processed_ds = processed_ds.map(remove_duplicate_frames, num_proc=num_proc)
    print("重复帧移除完成。")

    # 6. Create Input IDs (Combine text and audio tokens)
    print("\n步骤 6: 创建最终的 input_ids, labels, attention_mask...")
    final_ds = processed_ds.map(
        create_input_ids_map,
        batched=True,
        batch_size=100, # Can be larger for this step
        remove_columns=["text", "codes_list"], # Remove intermediate columns
        num_proc=num_proc
    )
    print("Input IDs 创建完成。")

     # Filter out any remaining empty sequences after final mapping
    print("\n步骤 7: 过滤空的最终样本...")
    original_count = len(final_ds)
    final_ds = final_ds.filter(lambda example: len(example["input_ids"]) > 0, num_proc=num_proc)
    filtered_count = len(final_ds)
    print(f"过滤完成。移除了 {original_count - filtered_count} 个空样本。")
    if filtered_count == 0:
        print("错误: 最终过滤后没有剩余的有效样本。")
        return


    print("\n--- 数据集准备完成 ---")
    print("最终数据集结构预览:")
    print(final_ds)
    print("\n第一个有效样本预览:")
    try:
        print(final_ds[0])
        print(f"Input IDs 长度: {len(final_ds[0]['input_ids'])}")
    except IndexError:
        print("无法预览第一个样本，最终数据集为空。")

    # 7. Push the FINAL PROCESSED dataset to Hugging Face Hub
    print("\n--- 推送处理后的数据集到 Hugging Face Hub ---")
    print("将尝试自动检测已登录的 Hugging Face 用户并直接推送。")
    print("确保你已通过 'huggingface-cli login' 登录。")

    hf_username = None
    try:
        hf_user = huggingface_hub.whoami()
        hf_username = hf_user.get('name')
        if not hf_username: raise ValueError("无法从 'whoami' 获取用户名。")
        print(f"检测到已登录用户: {hf_username}")
    except Exception as e:
         print(f"错误: 无法自动检测 Hugging Face 用户名: {e}")
         print("请确保你已通过 'huggingface-cli login' 成功登录。")

    # 准备数据集字典格式
    if isinstance(final_ds, datasets.Dataset):
        final_ds_dict = DatasetDict({"train": final_ds})
    else: # Should already be DatasetDict if split logic was added
        final_ds_dict = final_ds
        
    # 默认的本地保存路径
    save_path = "./processed_qjc_dataset"
    
    # 尝试推送到 Hugging Face Hub（如果有用户名）
    push_success = False
    if hf_username:
        repo_id = f"{hf_username}/{target_repo_name}"
        print(f"准备将处理后的数据集推送到: {repo_id}")
        
        try:
            print("正在推送处理后的数据集...")
            final_ds_dict.push_to_hub(repo_id, private=False)
            print(f"\n处理后的数据集成功推送到: https://huggingface.co/datasets/{repo_id}")
            print("\n现在你可以使用此仓库名配置 finetune/config.yaml:")
            print(f"dataset_name: \"{repo_id}\"")
            push_success = True
        except Exception as e:
            print(f"\n错误: 推送处理后的数据集失败: {e}")
            print("将自动尝试保存到本地...")
    
    # 如果推送失败或没有用户名，保存到本地
    if not push_success:
        print(f"\n保存处理后的数据集到本地: {save_path}")
        try:
            final_ds_dict.save_to_disk(save_path)
            print(f"处理后的数据集已成功保存到本地: {save_path}")
            print(f"你可以将此目录路径用于 finetune/config.yaml 的 dataset_name。")
            print(f"例如: dataset_name: \"{os.path.abspath(save_path)}\"")
        except Exception as e:
            print(f"错误: 保存到本地磁盘失败: {e}")
            print("请检查磁盘空间和写入权限。")
            sys.exit(1)


if __name__ == "__main__":
    main()