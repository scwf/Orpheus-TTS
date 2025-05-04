from datasets import load_dataset

# pip install datasets
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("canopylabs/zac-sample-dataset")

# 保存数据集
ds.save_to_disk("zac-sample-dataset")
