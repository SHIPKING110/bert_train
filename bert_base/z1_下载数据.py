from datasets import load_dataset

# 加载数据集
dataset_dict = load_dataset("NousResearch/hermes-function-calling-v1", cache_dir="data/")

# 获取 train 子集
train_dataset = dataset_dict["train"]

# 保存为 CSV
train_dataset.to_csv(r"D:\computer_soft\Microsoft_VS_Code\python_project\my_app\AI大模型应用开发\聚客AI\第五期\L2\day04-基于BERT模型的自定义微调训练\data\NousResearch___hermes-function-calling-v1\func_calling_singleturn\0.0.0\8f025148382537ba84cd325e1834b706e1461692\hermes-function-calling-v1.csv")
