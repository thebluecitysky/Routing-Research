import torch
import dgl
from dgl.nn import SAGEConv
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW
)

print(dgl.__version__)

# 尝试将图数据移动到 GPU
try:
    g = dgl.graph(([0, 1], [1, 2]))  # 创建一个简单的图
    g = g.to('cuda:0')  # 将图移动到 GPU
    print("DGL supports CUDA!")
except Exception as e:
    print(f"DGL does not support CUDA: {e}")

# 配置参数
LLM_NAME = '/data2/zhangtiant/DeepRouter/pretrained/distilbert'
SAVE_DIR = '/data2/zhangtiant/DeepRouter/saved'
BATCH_SIZE = 128
MAX_LENGTH = 1024
SEMANTIC_DIM = 768
SYNTACTIC_DIM = 128
FUSE_DIM = 256
NUM_CLASSES = 5
LR = 2e-5
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集（支持分批加载图数据）
class DualFeatureDataset(Dataset):
    def __init__(self, texts, graph_folder, labels, tokenizer, max_len, batch_size=100):
        self.texts = texts
        self.graph_folder = graph_folder  # 图文件所在的文件夹路径
        self.labels = [l-1 for l in labels]  # 标签1-5转0-4
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size  # 每批加载的图文件数量
        self.loaded_graphs = {}  # 用于缓存已加载的图数据
        self.current_batch_start = 0  # 当前批次的起始索引

    def __len__(self):
        return len(self.texts)

    def _load_graph_batch(self, start_idx):
        """加载一批图数据到内存中"""
        end_idx = min(start_idx + self.batch_size, len(self.texts))
        for idx in range(start_idx, end_idx):
            graph_path = os.path.join(self.graph_folder, f'graph_{idx}.dgl')
            self.loaded_graphs[idx] = dgl.load_graphs(graph_path)[0][0]  # 加载DGL图并缓存
        self.current_batch_start = start_idx

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 如果当前索引不在已加载的批次中，则加载新的批次
        if idx not in self.loaded_graphs:
            self._load_graph_batch(idx)
        
        # 从缓存中获取图数据
        graph = self.loaded_graphs[idx]

        # 句法特征提取
        if 'feat' not in graph.ndata:
            # 如果没有 'feat' 字段
            num_nodes = graph.num_nodes()
            graph.ndata['feat'] = torch.randn(num_nodes, 300)

        # 语义特征处理
        semantic_input = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': semantic_input['input_ids'].flatten(),
            'attention_mask': semantic_input['attention_mask'].flatten(),
            'dgl_graph': graph,
            'label': torch.tensor(label, dtype=torch.long)
        }

class DualChannelModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 语义通道
        self.roberta = DistilBertModel.from_pretrained(LLM_NAME)
        
        # 句法通道
        self.sage_conv1 = SAGEConv(300, 256, 'mean')  # 假设节点特征维度300
        self.sage_conv2 = SAGEConv(256, SYNTACTIC_DIM, 'mean')
        
        # 特征融合模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=SYNTACTIC_DIM,  # 128
            kdim=SYNTACTIC_DIM,       # 128
            vdim=SYNTACTIC_DIM,       # 128
            num_heads=4,
            batch_first=True
        )
        
        # 将 h_sem 的维度从 768 投影到 128
        self.projection = nn.Linear(SEMANTIC_DIM, SYNTACTIC_DIM)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(SEMANTIC_DIM + SYNTACTIC_DIM, FUSE_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSE_DIM, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, dgl_graph):
        # 语义特征提取
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h_sem = outputs.last_hidden_state[:, 0, :]  # [CLS]向量，维度 [BS, 768]
        
        # 句法特征提取
        features = dgl_graph.ndata['feat'].float()  # 假设节点特征存储在'feat'字段
        x = self.sage_conv1(dgl_graph, features)
        x = torch.relu(x)
        h_syn = self.sage_conv2(dgl_graph, x)
        # 将 h_syn 存储在图的节点数据中
        dgl_graph.ndata['h_syn'] = h_syn

        # 使用 dgl.mean_nodes 计算全局平均
        h_syn_global = dgl.mean_nodes(dgl_graph, 'h_syn')  # 全局池化，维度 [BS, 128]
        
        # 将 h_sem 的维度从 768 投影到 128
        h_sem_proj = self.projection(h_sem)  # [BS, 128]
        
        # 特征融合
        attn_output, _ = self.cross_attn(
            query=h_sem_proj.unsqueeze(1),  # [BS, 1, 128]
            key=h_syn_global.unsqueeze(1),  # [BS, 1, 128]
            value=h_syn_global.unsqueeze(1) # [BS, 1, 128]
        )
        
        # 拼接融合
        fused = torch.cat([
            h_sem,  # [BS, 768]
            attn_output.squeeze(1)  # [BS, 128]
        ], dim=1)  # [BS, 768 + 128]
        
        # 分类预测
        logits = self.classifier(fused)
        return logits

# 训练流程
def train():
    tokenizer = DistilBertTokenizer.from_pretrained(LLM_NAME)
    dataset = pd.read_csv('/data2/zhangtiant/DeepRouter/data/combined_data.csv')

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    graph_folder = '/data2/zhangtiant/DeepRouter/data/graph'
    train_set = DualFeatureDataset(
        [dataset['prompt'][i] for i in train_dataset.indices],
        graph_folder,
        [dataset['score'][i] for i in train_dataset.indices],
        tokenizer,
        MAX_LENGTH,
        batch_size=100
    )
    val_set = DualFeatureDataset(
        [dataset['prompt'][i] for i in val_dataset.indices],
        graph_folder,
        [dataset['score'][i] for i in val_dataset.indices],
        tokenizer,
        MAX_LENGTH,
        batch_size=100
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=custom_collate, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate, shuffle=False)

    model = DualChannelModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 用于保存最佳模型
    best_val_loss = float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 用于可视化
    train_losses = []
    val_losses = []

    from tqdm import tqdm  # 导入 tqdm

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        # 使用 tqdm 包装 train_loader
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch in train_loader_tqdm:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            graphs = batch['dgl_graph'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(input_ids, attention_mask, graphs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 更新 tqdm 的描述信息，显示当前 loss
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证集
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            # 使用 tqdm 包装 val_loader
            val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_loader_tqdm:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                graphs = batch['dgl_graph'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                logits = model(input_ids, attention_mask, graphs)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                # 更新 tqdm 的描述信息，显示当前 loss
                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    # 可视化 loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))
    plt.show()

# 自定义数据整理函数
def custom_collate(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'dgl_graph': dgl.batch([x['dgl_graph'] for x in batch]),
        'label': torch.stack([x['label'] for x in batch])
    }

if __name__ == "__main__":
    train()