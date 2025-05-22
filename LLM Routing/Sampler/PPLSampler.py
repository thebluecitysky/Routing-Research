import numpy as np
import pandas as pd
from scipy.stats import beta
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 困惑度计算模块
class PerplexityCalculator:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def calculate_ppl(self, text, stride=512):
        """计算单个文本的困惑度"""
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        for begin_index in range(0, seq_len, stride):
            end_index = min(begin_index + max_length, seq_len)
            input_ids = encodings.input_ids[:, begin_index:end_index].to(self.device)
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

# 2. 动态采样策略核心类
class DynamicSampler:
    def __init__(self, alpha=1.5, beta_params=(2,5)):
        self.alpha = alpha          # 难度权重系数
        self.beta_a, self.beta_b = beta_params  # Beta分布参数
    
    def _assign_difficulty_levels(self, ppl_series):
        """使用四分位数划分难度等级"""
        quantiles = ppl_series.quantile([0.25, 0.5, 0.75])
        bins = [-np.inf, quantiles[0.25], quantiles[0.5], quantiles[0.75], np.inf]
        labels = ['L1', 'L2', 'L3', 'L4']
        return pd.cut(ppl_series, bins=bins, labels=labels)
    
    def _calculate_sampling_weights(self, df):
        """计算归一化采样权重"""
        weighted_ppl = np.power(df['ppl'], self.alpha)
        total = weighted_ppl.sum()
        return weighted_ppl / total
    
    def _beta_sampling_factor(self, difficulty_level):
        """根据难度等级生成Beta分布采样因子"""
        level_map = {'L1':0.1, 'L2':0.3, 'L3':0.7, 'L4':0.9}  # 各等级基准值
        base = level_map[difficulty_level]
        return beta.ppf(base, self.beta_a, self.beta_b)
    
    def resample_data(self, df, target_size):
        """执行动态重采样"""
        # 计算基础权重
        df['base_weight'] = self._calculate_sampling_weights(df)
        
        # 应用Beta分布调整
        df['beta_factor'] = df['difficulty_level'].apply(self._beta_sampling_factor)
        df['final_weight'] = df['base_weight'] * df['beta_factor']
        df['final_weight'] /= df['final_weight'].sum()  # 重新归一化
        
        # 分层采样
        sampled_df = df.sample(
            n=target_size,
            weights='final_weight',
            replace=True,   # 允许过采样
            random_state=42
        ).reset_index(drop=True)
        
        return sampled_df

# 3. 完整处理流程
def main_process(input_path, output_path, target_size=200000):
    # 加载数据
    df = pd.read_csv(input_path)
    print(f"原始数据量：{len(df)} 条")
    
    # 步骤1：计算困惑度
    ppl_calculator = PerplexityCalculator()
    df['ppl'] = df['prompt'].progress_apply(ppl_calculator.calculate_ppl)  # 使用tqdm进度条
    
    # 步骤2：划分难度等级
    sampler = DynamicSampler(alpha=1.5)
    df['difficulty_level'] = sampler._assign_difficulty_levels(df['ppl'])
    
    # 步骤3：动态重采样
    resampled_df = sampler.resample_data(df, target_size)
    
    # 保存结果
    resampled_df.to_csv(output_path, index=False)
    print(f"采样后数据量：{len(resampled_df)} 条")
    
    # 打印分布统计
    dist = resampled_df['difficulty_level'].value_counts(normalize=True)
    print("\n难度分布：")
    print(dist.sort_index())

# 示例执行
if __name__ == "__main__":
    input_file = "original_data.csv"
    output_file = "resampled_data.csv"
    main_process(input_file, output_file)