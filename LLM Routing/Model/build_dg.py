import spacy
import dgl
import os
import pandas as pd
from tqdm import tqdm
import torch

nlp = spacy.load("en_core_web_sm")

def build_dependency_graph(text, graph_id, save_dir="/data2/zhangtiant/DeepRouter/data/graph"):
    """构建依存语法图并保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    doc = nlp(text)
    edges = []
    for token in doc:
        if token.head.i != token.i:  # 排除自环
            edges.append((token.head.i, token.i))

    # 创建DGL图
    src_nodes = [s for s, d in edges]
    dst_nodes = [d for s, d in edges]
    g = dgl.graph((src_nodes, dst_nodes))
    
    # 为每个节点添加特征（假设特征维度为300）
    num_nodes = g.num_nodes()
    node_features = torch.randn(num_nodes, 300)  # 随机初始化节点特征
    g.ndata['feat'] = node_features  # 将特征存储在 'feat' 字段中

    # 保存图
    dgl.save_graphs(os.path.join(save_dir, f"{graph_id}.dgl"), [g])

# 示例使用
if __name__ == "__main__":

    # 读取 combined_data.csv
    df = pd.read_csv('/data2/zhangtiant/DeepRouter/data/combined_data.csv')

    # 使用 tqdm 遍历每一行的 prompt 并调用 build_dependency_graph
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt = row['prompt']
        graph_id = f"graph_{idx}"  # 使用索引作为图的唯一 ID
        build_dependency_graph(prompt, graph_id)

    print("All prompts processed and graphs saved!")