# train_val.py

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from MyDate import MyDataset
from nete import Model

# ===== 设置设备信息 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 12  # 最多训练轮数
PATIENCE = 5   # early stopping 的容忍轮数

# ===== 加载分词器 =====
token = BertTokenizer.from_pretrained(
    r".\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
)

# ===== 数据预处理函数 =====
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, label

# ===== 创建数据集和加载器 =====
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

val_dataset = MyDataset("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

# ===== 训练主程序 =====
if __name__ == '__main__':
    print("设备：", DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(__file__), "params")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_val_acc = 0.0
    no_improve_epochs = 0
    start_epoch = 0

    # ===== 尝试加载断点 =====
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_acc = checkpoint["best_val_acc"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ 成功加载断点模型，继续从第 {start_epoch+1} 轮开始训练...")
    else:
        print("🔁 未检测到断点模型，从头开始训练")

    # ===== 训练过程 =====
    for epoch in range(start_epoch, EPOCH):
        model.train()
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                pred = out.argmax(dim=1)
                acc = (pred == label).sum().item() / len(label)
                print(f"[Epoch {epoch+1}] Step {i} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

        # ===== 验证阶段 =====
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, label in val_loader:
                input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
                out = model(input_ids, attention_mask, token_type_ids)
                val_loss += loss_func(out, label)
                pred = out.argmax(dim=1)
                val_acc += (pred == label).sum().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"[Epoch {epoch+1}] Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # ===== 保存最优模型 =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_bert.pth"))
            print(f"✅ Epoch {epoch+1}: 保存最优参数 acc={best_val_acc:.4f}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ 验证集未提升，已等待 {no_improve_epochs}/{PATIENCE} 轮")

        # ===== 保存最后一轮模型 =====
        torch.save(model.state_dict(), os.path.join(save_dir, "last_bert.pth"))
        print(f"💾 Epoch {epoch+1}: 最后一轮参数保存成功")

        # ===== 保存断点状态 =====
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'no_improve_epochs': no_improve_epochs
        }, checkpoint_path)
        print(f"📌 断点已保存，可断电续训")

        # ===== EarlyStopping =====
        if no_improve_epochs >= PATIENCE:
            print(f"⛔ EarlyStopping 触发，连续 {PATIENCE} 轮验证集未提升，训练终止")
            break
