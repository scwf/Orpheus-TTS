from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
max_length = config.get("max_length", 4096)  # 默认为4096，如果未在配置中指定
gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)  # 默认为4

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# 启用梯度检查点以节省内存
model.gradient_checkpointing_enable()
model.half()

# 加载数据集
ds = load_dataset(dsn, split="train") 

# 裁剪长序列以节省内存
def preprocess_function(examples):
    # 裁剪输入序列
    for key in ['input_ids', 'labels', 'attention_mask']:
        if key in examples:
            for i in range(len(examples[key])):
                if len(examples[key][i]) > max_length:
                    examples[key][i] = examples[key][i][:max_length]
    return examples

# 应用预处理
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    desc="裁剪长序列"
)

wandb.init(project=project_name, name = run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="wandb", 
    save_steps=save_steps,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
    # 添加内存优化设置
    gradient_accumulation_steps=gradient_accumulation_steps,  # 累积梯度减少内存使用
    gradient_checkpointing=True,    # 启用梯度检查点
    optim="adamw_torch_fused",      # 使用融合的 AdamW 优化器
    max_grad_norm=1.0,              # 梯度裁剪
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_ds,  # 使用处理后的数据集
)

trainer.train()

