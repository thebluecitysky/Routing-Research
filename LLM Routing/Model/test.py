import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 定义模型和路由策略
MODELS = {
    "DeepSEEK-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "LLaMA-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "BERT": "bert-base-uncased",
    "AUTOROUTE": "path_to_autoroute_model",  # 你的路由策略模型
    "Random": None,  # 随机路由策略
}

# 加载测试数据集
def load_test_data(dataset_name):
    if dataset_name == "MMLU":
        return load_dataset("mmlu", "high_school")  # 加载 MMLU 数据集
    elif dataset_name == "MTBench":
        return load_dataset("mt_bench")  # 加载 MTBench 数据集
    elif dataset_name == "GSM8K":
        return load_dataset("gsm8k")  # 加载 GSM8K 数据集
    else:
        raise ValueError("Unsupported dataset")

# 模型推理函数
def model_inference(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 更新路由策略函数
def route_question(question, route_strategy="AUTOROUTE", route_model=None, tokenizer=None, threshold=0.5):
    """
    根据路由策略决定使用哪个模型
    :param question: 输入的问题
    :param route_strategy: 路由策略 (AUTOROUTE, Random, DeepSEEK-32B, LLaMA-8B)
    :param route_model: 路由模型 (DualChannelModel 实例)
    :param tokenizer: 路由模型对应的分词器
    :param threshold: 阈值，用于判断是否使用小模型
    :return: 选择的模型名称
    """
    if route_strategy == "Random":
        # 随机路由策略
        return np.random.choice(["DeepSEEK-32B", "LLaMA-8B"])
    elif route_strategy in ["DeepSEEK-32B", "LLaMA-8B"]:
        # 固定使用某个模型
        return route_strategy
    elif route_strategy == "AUTOROUTE" and route_model is not None:
        # 使用路由模型进行推理
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # 获取图结构 (假设我们有一个预处理好的图结构 dgl_graph)
        from build_dg import build_dependency_graph
        dgl_graph = build_dependency_graph(question)  # 假设函数 preprocess_question_to_graph 存在
        
        # 模型推理
        with torch.no_grad():
            logits = route_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                dgl_graph=dgl_graph
            )
        
        # 计算第 4 类和第 5 类 logits 的总和
        high_class_logits_sum = torch.sum(logits[:, [3, 4]], dim=1).item()  # 索引从 0 开始，第 4 类和第 5 类是索引 3 和 4
        
        # 根据阈值选择模型
        if high_class_logits_sum > threshold:
            return "LLaMA-8B"  # 小模型
        else:
            return "DeepSEEK-32B"  # 大模型
    else:
        raise ValueError(f"Unknown routing strategy: {route_strategy}")

# 打分函数（用于 MTBench）
def score_answer(answer, tokenizer, reference_answer):
    # 使用 DeepSEEK 32B 对答案进行打分（假设打分范围为 0-10）
    prompt = f"Rate the following answer on a scale of 0 to 10:\nAnswer: {answer}\nReference: {reference_answer}\nScore:"
    score = model_inference(MODELS["DeepSEEK-32B"], tokenizer, prompt)
    return float(score)

# 评测函数
def evaluate_model(model_name, dataset_name):
    # 如果是路由策略，我们需要加载两个模型
    if model_name in ["AUTOROUTE", "Random"]:
        # 预加载两个模型
        tokenizers = {
            "DeepSEEK-32B": AutoTokenizer.from_pretrained(MODELS["DeepSEEK-32B"]),
            "LLaMA-8B": AutoTokenizer.from_pretrained(MODELS["LLaMA-8B"]),
        }
        models = {
            "DeepSEEK-32B": AutoModelForCausalLM.from_pretrained(MODELS["DeepSEEK-32B"]).to(device),
            "LLaMA-8B": AutoModelForCausalLM.from_pretrained(MODELS["LLaMA-8B"]).to(device),
        }
    else:
        # 加载单个模型和分词器
        model_path = MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # 加载数据集
    dataset = load_test_data(dataset_name)
    results = []
    model_usage = {"DeepSEEK-32B": 0, "LLaMA-8B": 0}  # 记录模型使用情况

    for example in tqdm(dataset, desc=f"Evaluating {model_name} on {dataset_name}"):
        prompt = example["question"]
        reference_answer = example.get("answer", None)

        # 根据路由策略选择模型
        if model_name in ["AUTOROUTE", "Random"]:
            selected_model = route_question(prompt, model_name)
            model_usage[selected_model] += 1
            answer = model_inference(models[selected_model], tokenizers[selected_model], prompt)
        else:
            # 直接使用指定模型
            answer = model_inference(model, tokenizer, prompt)

        if dataset_name == "MTBench":
            # 对 MTBench 数据集进行打分
            score = score_answer(answer, tokenizer, reference_answer)
            results.append(score)
        else:
            # 对 MMLU 和 GSM8K 数据集进行正确性判断
            is_correct = (answer.strip().lower() == reference_answer.strip().lower())
            results.append(int(is_correct))

    # 打印模型使用情况（如果是路由策略）
    if model_name in ["AUTOROUTE", "Random"]:
        total = sum(model_usage.values())
        print(f"Model usage for {model_name}:")
        for m, count in model_usage.items():
            print(f"  {m}: {count} ({count/total*100:.1f}%)")

    # 计算性能指标
    if dataset_name == "MTBench":
        avg_score = np.mean(results)
        print(f"{model_name} on {dataset_name}: Average Score = {avg_score:.2f}")
    else:
        accuracy = np.mean(results) * 100
        print(f"{model_name} on {dataset_name}: Accuracy = {accuracy:.2f}%")

# 计算 CPT 和 APGR 指标
def calculate_cpt_apgr(results):
    sorted_results = sorted(results)
    n = len(sorted_results)
    cpt_50 = sorted_results[int(n * 0.5)]
    cpt_80 = sorted_results[int(n * 0.8)]
    apgr = np.mean(sorted_results)
    return cpt_50, cpt_80, apgr

def evaluate_mmlu_subset(model_name, subset_name):
    # 如果是路由策略，我们需要加载两个模型
    if model_name in ["AUTOROUTE", "Random"]:
        # 预加载两个模型
        tokenizers = {
            "DeepSEEK-32B": AutoTokenizer.from_pretrained(MODELS["DeepSEEK-32B"]),
            "LLaMA-8B": AutoTokenizer.from_pretrained(MODELS["LLaMA-8B"]),
        }
        models = {
            "DeepSEEK-32B": AutoModelForCausalLM.from_pretrained(MODELS["DeepSEEK-32B"]).to(device),
            "LLaMA-8B": AutoModelForCausalLM.from_pretrained(MODELS["LLaMA-8B"]).to(device),
        }
    else:
        # 加载单个模型和分词器
        model_path = MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # 加载 MMLU 子集数据
    subset_path = f"mmlu_high_school_{subset_name}.csv"
    subset_data = pd.read_csv(subset_path)
    results = []
    model_usage = {"DeepSEEK-32B": 0, "LLaMA-8B": 0}  # 记录模型使用情况

    for _, row in tqdm(subset_data.iterrows(), desc=f"Evaluating {model_name} on {subset_name}"):
        prompt = row["question"]
        reference_answer = row["answer"]

        # 根据路由策略选择模型
        if model_name in ["AUTOROUTE", "Random"]:
            selected_model = route_question(prompt, model_name)
            model_usage[selected_model] += 1
            answer = model_inference(models[selected_model], tokenizers[selected_model], prompt)
        else:
            # 直接使用指定模型
            answer = model_inference(model, tokenizer, prompt)

        is_correct = (answer.strip().lower() == reference_answer.strip().lower())
        results.append(int(is_correct))

    # 打印模型使用情况（如果是路由策略）
    if model_name in ["AUTOROUTE", "Random"]:
        total = sum(model_usage.values())
        print(f"Model usage for {model_name} on {subset_name}:")
        for m, count in model_usage.items():
            print(f"  {m}: {count} ({count/total*100:.1f}%)")

    # 计算 CPT 和 APGR
    cpt_50, cpt_80, apgr = calculate_cpt_apgr(results)
    print(f"{model_name} on {subset_name}: CPT (50%) = {cpt_50:.2f}, CPT (80%) = {cpt_80:.2f}, APGR = {apgr:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载路由模型
    ROUTE_MODEL_PATH = "path_to_autoroute_model"
    route_tokenizer = AutoTokenizer.from_pretrained(ROUTE_MODEL_PATH)
    from train import DualChannelModel
    route_model = DualChannelModel().to(device)
    route_model.load_state_dict(torch.load(ROUTE_MODEL_PATH, map_location=device))
    route_model.eval()

    # 对比实验
    for model_name in MODELS.keys():
        for dataset_name in ["MMLU", "MTBench", "GSM8K"]:
            evaluate_model(model_name, dataset_name, route_model=route_model, tokenizer=route_tokenizer)

    # MMLU 特定子集评测
    for subset_name in ["computer_science", "mathematics", "macroeconomics", "world_history"]:
        evaluate_mmlu_subset("DeepSEEK-32B", subset_name, route_model=route_model, tokenizer=route_tokenizer)
        evaluate_mmlu_subset("LLaMA-8B", subset_name, route_model=route_model, tokenizer=route_tokenizer)
        evaluate_mmlu_subset("AUTOROUTE", subset_name, route_model=route_model, tokenizer=route_tokenizer)