# train_val.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer
from MyDate import MyDataset
from nete import Model
from collections import Counter
from torch.cuda import amp  # 导入混合精度训练

# ===== 设置设备信息 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 12  # 最多训练轮数
PATIENCE = 5   # early stopping 的容忍轮数
LEARNING_RATE = 2e-5  # BERT推荐学习率
WEIGHT_DECAY = 0.01   # L2正则化强度

# ===== 加载分词器 =====
token = BertTokenizer.from_pretrained(
    r"D:\computer_soft\Microsoft_VS_Code\python_project\my_app\AI大模型应用开发\聚客AI\第五期\L2\day04-基于BERT模型的自定义微调训练\demo_04\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
)

# ===== 数据预处理函数 =====
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=128,  # 减小序列长度（重要！512->128）
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
val_dataset = MyDataset("validation")

# 检查数据集
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

# 检查标签分布
train_labels = [label for _, label in train_dataset]
val_labels = [label for _, label in val_dataset]

print("训练集标签分布:", Counter(train_labels))
print("验证集标签分布:", Counter(val_labels))

# 减小批次大小以适应显存
BATCH_SIZE = 8  # 原为50，大幅减小以适应4GB显存

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn
)

# ===== 训练主程序 =====
if __name__ == '__main__':
    print("设备：", DEVICE)
    # 创建模型（只解冻最后2层）
    model = Model(dropout_rate=0.3, unfreeze_last_n_layers=2).to(DEVICE)
    
    # 使用带权重衰减的AdamW优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化需要梯度的参数
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * EPOCH,  # 总迭代次数
        eta_min=1e-6  # 最小学习率
    )
    
    # 创建混合精度训练的GradScaler
    scaler = amp.GradScaler()

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
        total_train_samples = 0
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            batch_size = input_ids.size(0)
            total_train_samples += batch_size
            
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)

            # 使用混合精度训练（节省显存）
            with amp.autocast():
                out = model(input_ids, attention_mask, token_type_ids)
                loss = torch.nn.functional.cross_entropy(out, label)

            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪防止爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()  # 更新学习率

            # 计算训练准确率
            pred = out.argmax(dim=1)
            acc = (pred == label).sum().item() / batch_size
            
            train_loss_sum += loss.item() * batch_size
            train_acc_sum += (pred == label).sum().item()

            if i % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch+1}] Step {i} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | LR: {current_lr:.2e}")

        # 计算epoch平均训练损失和准确率
        train_loss = train_loss_sum / total_train_samples
        train_acc = train_acc_sum / total_train_samples
        print(f"[Epoch {epoch+1}] Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        # ===== 验证阶段 =====
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, label in val_loader:
                batch_size = input_ids.size(0)
                total_val_samples += batch_size
                
                input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
                out = model(input_ids, attention_mask, token_type_ids)
                
                # 计算验证损失
                batch_loss = torch.nn.functional.cross_entropy(out, label)
                val_loss += batch_loss.item() * batch_size
                
                # 计算验证准确率
                pred = out.argmax(dim=1)
                val_acc += (pred == label).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss /= total_val_samples
        val_acc /= total_val_samples
        
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

        # 检查准确率是否异常
        if val_acc < 0.5:  # 如果准确率低于50%
            print(f"⚠️ 警告：验证集准确率异常低 ({val_acc:.4f})，请检查数据或模型")

        # ===== 保存最后一轮模型 =====
        torch.save(model.state_dict(), os.path.join(save_dir, "last_bert.pth"))
        print(f"💾 Epoch {epoch+1}: 最后一轮参数保存成功")

        # ===== 保存断点状态 =====
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'no_improve_epochs': no_improve_epochs
        }, checkpoint_path)
        print(f"📌 断点已保存，可断电续训")

        # ===== EarlyStopping =====
        if no_improve_epochs >= PATIENCE:
            print(f"⛔ EarlyStopping 触发，连续 {PATIENCE} 轮验证集未提升，训练终止")
            break
