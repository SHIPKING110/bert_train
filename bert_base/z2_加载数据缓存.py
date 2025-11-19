from datasets import load_dataset

# 加载你自己保存的 CSV 文件
dataset = load_dataset(
    "csv",
    data_files=r".\data\hermes-function-calling-v1.csv"
)

print(dataset)

train_data = dataset["train"]
for data in train_data:
    print(data)