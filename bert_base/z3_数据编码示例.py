from transformers import BertTokenizer

# 加载训练模型
token = BertTokenizer.from_pretrained(
    r".\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
)

# 准备要编码的文本数据
sents = [
    "床前明月光，",
    "我是要成为海贼王的男人，额谷歌阶级"
]

# 批量编码
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    truncation=True,
    max_length=15,
    padding="max_length",
    return_tensors=None,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,  # ← 这里要加 s
    return_length=True
)

# 输出各字段
for i, k in out.items():
    print(i, ":", k)

# 解码输出
print(token.decode(out["input_ids"][0]))
print(token.decode(out["input_ids"][1]))
