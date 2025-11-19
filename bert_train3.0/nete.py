import torch
from transformers import BertModel, BertConfig

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Model(torch.nn.Module):
    def __init__(self, dropout_rate=0.3, unfreeze_last_n_layers=2):
        super().__init__()
        # 加载配置而不是整个模型（节省内存）
        config = BertConfig.from_pretrained(
            r".\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
        )
        
        # 启用梯度检查点（显著减少显存使用）
        config.gradient_checkpointing = True
        
        # 加载模型（使用配置）
        self.bert = BertModel.from_pretrained(
            r".\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f",
            config=config
        ).to(DEVICE)
        
        # 冻结所有BERT参数（默认）
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 只解冻最后几层（节省显存同时保持一定微调能力）
        for layer in self.bert.encoder.layer[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # 添加Dropout层防止过拟合
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768, 2)
    
    # 使用模型处理数据（执行前向计算）
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取BERT输出（注意：不使用池化输出，因为它需要额外内存）
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用第一个token的隐藏状态（CLS token）
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 应用Dropout
        cls_output = self.dropout(cls_output)
        
        # 通过全连接层分类
        out = self.fc(cls_output)
        return out