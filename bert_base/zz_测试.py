from datasets import load_dataset,load_from_disk


# 加载缓存数据
datasets = load_from_disk(r".\ChnSentiCorp")
print(datasets)

