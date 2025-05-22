import pandas as pd
import json
import numpy as np
# 读取 train.jsonl 文件
gpt4_data = []
with open('/data2/zhangtiant/DeepRouter/data/LLM-judge-labeled datasets/train.jsonl', 'r') as f:
    for line in f:
        gpt4_data.append(json.loads(line))

# 将数据转换为 DataFrame
df = pd.DataFrame(gpt4_data)

# 创建新的 DataFrame
new_df = pd.DataFrame()

# 填充新 DataFrame 的字段
new_df['id'] = df.index
new_df['model_a'] = 'gpt-4'
new_df['model_b'] = 'mixtral-8x7b-instruct-v0.1'
new_df['prompt'] = df['prompt']
new_df['response_a'] = df['gpt4_response']
new_df['response_b'] = df['mixtral_response']

# 根据 mixtral_score 设置 winner_model_a, winner_model_b, winner_tie
new_df['winner_model_a'] = (df['mixtral_score'] < 3).astype(int)
new_df['winner_model_b'] = (df['mixtral_score'] > 3).astype(int)
new_df['winner_tie'] = (df['mixtral_score'] == 3).astype(int)
new_df['score'] = df['mixtral_score']

prefer_data = pd.read_csv('/data2/zhangtiant/DeepRouter/data/train.csv')
# 计算 score
def calculate_score(row):
    if row['winner_model_a'] == 1:
        return np.random.choice([1, 2])  # 随机选择 1 或 2
    elif row['winner_tie'] == 1:
        return 3
    else:
        return np.random.choice([4, 5])  # 随机选择 4 或 5
# 应用计算函数
prefer_data['score'] = prefer_data.apply(calculate_score, axis=1)

# 合并 prefer_data 和 new_df
combined_df = pd.concat([prefer_data, new_df], ignore_index=True)

# 重新计算 id
combined_df['id'] = combined_df.index

combined_df.to_csv('/data2/zhangtiant/DeepRouter/data/combined_data.csv', index=False)

