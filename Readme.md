# Research on Efficient Reasoning of Large Language Model Based on Routing Strategy 

Implementation of LLM Routing,Graduation Project of City University of Macau.

Existing large language model routing strategies focus on resource allocation optimization, while our RAGRouter framework optimizes the output space of semantic-syntactic feature alignment and knowledge enhancement through RAG vector retrieval enhancement technology.

If you want to know more about the experiment, see RAGRouter test.ipynb in Experiment Log.

![image1](https://github.com/thebluecitysky/-/blob/master/png/image1.png)



Our core features include:

1.  Utilizes DeBERTa-v3 to extract fine-grained semantic features from user queries; Leverages GraphSAGE Graph Neural Network to analyze dependency syntax structures and obtain syntactic feature representations; Applies a cross-modal attention mechanism to dynamically integrate semantic and syntactic features, forming a comprehensive query representation that enhances the model's ability to assess task difficulty.
    
2.  Retrieval-Augmented Generation (RAG) Mechanism for Routing Decisions: Constructs a vector knowledge base using historical queries and routing labels; Employs the K-Nearest Neighbors algorithm to retrieve historically similar queries based on the input query; Generates context-enhanced prompts from retrieved results to guide the fine-tuning of the QWEN 7B model, improving the precision and effectiveness of model selection.
    
3.  Validated Effectiveness and Superior Real-World Deployment Performance: Demonstrates the model's effectiveness through
    experiments on public benchmark datasets;Improves routing accuracy and stability, while reducing the frequency of large-scale model invocation without compromising response quality;

The purpose of the large language model routing strategy model is to optimize the balance between cost and response quality without reducing inference performance.

![image2](https://github.com/thebluecitysky/-/blob/master/png/image2.png)



![image3](https://github.com/thebluecitysky/-/blob/master/png/image3.png)

As shown in the two figures above, the routing strategy of the large language model------RouteLLM can dynamically select a large model with larger or smaller parameters to answer the question based on the difficulty of the user\'s request.

![image4](https://github.com/thebluecitysky/-/blob/master/png/image4.png)

As shown in the above figure, we build a RAG vector database to implement vector retrieval enhancement to optimize the training of the large language model routing strategy model. For each question entered by the user, by retrieving questions with similar meanings and their routing labels in the vector knowledge base, we can generate context enhancement prompts to guide the fine-tuning of the large model.

## Requirements

### For Four routing strategy models for routellm(include BERT, Causal LLM,Matrix Factorization, SW Ranking)

#### (1)Basic environment requirements:

Python \>= 3.8

PyTorch \>= 2.0.1

transformers \>= 4.40.0

numpy \>= 1.24.0

pandas \>= 2.0.0

#### \(2\) Framework core dependencies

litellm \>= 1.40.2

fastapi \>= 0.110.

uvicorn \>= 0.27.04

httpx \>= 0.27.04

huggingface-hub \>= 0.22.24

datasets \>= 2.18.04

sentence-transformers \>= 3.0.0

scikit-learn \>= 1.3.2

scipy \>= 1.11.4

tqdm \>= 4.66.1

## Install the complete environment with the following command

```python
# 基础环境（推荐使用 Python 3.10）

conda create -n routellm python=3.10

conda activate routellm

\# 安装核心框架 + 服务器 + 评估组件

pip install \"routellm\[serve,eval\]\"

\# 安装 PyTorch（根据 CUDA 版本选择，以下为 CPU 版示例）

pip install torch==2.4.1

\# 按需安装模型提供商 SDK

pip install openai anthropic cohere
```

Verify Installation:

```python
python -c \"from routellm import Controller; 

print(\'Install success\')\"
```



### For RAGRouter model

python==3.8.10

torch==2.1.2 \# PyTorch

transformers==4.36.0

sentencepiece==0.1.99

torch-geometric==2.4.0

faiss-cpu==1.7.4

numpy==1.23.5

pandas==1.5.3

tqdm==4.64.1

#### accelerate==0.24.1

datasets==2.15.0

spacy==3.7.2

spacy-legacy==3.0.12

spacy-loggers==1.0.5

en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl



## Key Dependency Description

#### (1) Semantic Feature Extraction

DeBERTa-v3 Load the microsoft/deberta-v3-base model via the transformers library

#### (2) Syntactic feature extraction

GraphSAGE: Depends on torch-geometric,additional dependencies need to be installed.

pip install torch-scatter torch-sparse torch-cluster-f <https://data.pyg.org/whl/torch-2.1.0+cu121.html>.

#### (3) Cross-modal attention fusion

Custom CM-MHA modules need to be implemented based on torch.nn.MultiheadAttention, without additional dependencies

#### (4) RAG Vector Library

Faiss: Used for efficient similarity retrieval (GPU accelerated version requires installation of faiss-gpu)

#### (5) Model fine-tuning

Full parameter fine-tuning Qwen-7B requires at least 24GB GPU.If the memory is insufficient, you can use QLoRA quantization
fine-tuning in combination with bitsandbytes + peft.

#### (6) Syntactic analysis

Dependency parsing relies on spacy\'s English model en_core_web_sm.



#### Installation command

```
# 1. 创建虚拟环境 conda create -n ragrouter python=3.8.10 conda activate ragrouter

\# 2. 安装PyTorch（CUDA 12.1） pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \--index-url
<https://download.pytorch.org/whl/cu121>

\# 3. 安装其他依赖 pip install -r requirements.txt

\# 4. 安装spaCy英文模型 python -m spacy download en_core_web_sm
```



## Experimental Results

Before officially starting the experiments in Table 1 to Table 6, we must first refer to the contents of the uploaded file
[DataAugmentation\] to perform data augmentation on the training data and data sampling processing in the file \[Sample\]. Then, we build a vector database through rag.py in the file \[Model\] and fine-tune the Qianwen 7B large model through qwen_train.py to complete the training of the RAGRouter routing strategy model for the next step.Also, our ablation experiments are all conducted on the MMLU test set.

## Routers on MMLU and GSM8K

You can create the script router_strategy_comparison.py according to the uploaded code file to compare the performance differences between RAGRouter and the baseline routing strategy on the MMLU/GSM8K dataset.

```python
python router_strategy_comparison.py 

--datasets MMLU GSM8K # 测试数据集

--strategies RAGRouter SW_Ranking BERT Matrix_Factorization Causal_LLM
Random

--strong_model DeepSeek-32B # 强模型选择

--weak_model LLAMA-8B # 弱模型选择

--output_dir . # 结果保存路径
```



Table 1:Routers on MMLU

| Model                | CPT(50%) | CPT（80%) | APGR  |
| -------------------- | -------- | --------- | ----- |
| BERT                 | 45.86%   | 75.33%    | 0.527 |
| Causal LLM           | 42.99%   | 74.60%    | 0.545 |
| Matrix Factorization | 40.80%   | 74.46%    | 0.542 |
| SW Ranking           | 46.83%   | 76.43%    | 0.550 |
| Random               | 50.05%   | 79.95%    | 0.498 |
| RAGRouter            | 37.25%   | 71.89%    | 0.589 |

Table 2:Routers on GSM8K

| Model                 | CPT(50%) | CPT(80%) | APGR  |
| --------------------- | -------- | -------- | ----- |
| BERT                  | 44.23%   | 78.69%   | 0.539 |
| Causal LLM            | 32.82%   | 62.35%   | 0.631 |
| Matrix  Factorization | 38.35%   | 72.23%   | 0.571 |
| SW Ranking            | 40.78%   | 71.89%   | 0.559 |
| Random                | 50.03%   | 80.04%   | 0.496 |
| RAGRouter             | 28.21%   | 57.69%   | 0.695 |





## Experimental comparison of multi-source data fusion with and without data enhancement

You can create the script ablation_ragrouter.py according to the uploaded LLM Routing code document to quickly conduct ablation experiments. In the following example, original represents the enhanced data and original represents the original data.

```python
python ablation_ragrouter.py

--train_data ./data/original 

--enhanced_data ./data/enhanced
```

Table 3

| Model                                    | CPT（50%） | CPT（80%） | APGR  |
| ---------------------------------------- | ---------- | ---------- | ----- |
| No multi-source data integration combine | 38.35%     | 72.56%     | 0.572 |
| Fusion of multi-source data              | 37.25%     | 71.89%     | 0.589 |



## Experimental comparison with and without data sampling

You can create the script ablation_sampling.py according to the uploaded LLM Routing code document to quickly conduct ablation experiments. In the following example content, sampled represents the sampled data and original represents the original data.

```python
python ablation_sampling.py 

--raw_data ./data/original 

--sampled_data ./data/sampled
```

Table 4

| Model                 | CPT（50%） | CPT（80%） | APGR  |
| --------------------- | ---------- | ---------- | ----- |
| No data sampling      | 38.15%     | 72.32%     | 0.576 |
| Conduct data sampling | 37.25%     | 71.89%     | 0.589 |



## Experimental comparison of semantic and syntactic feature extraction and non-extraction

You can create the script ablation_features.py according to the uploaded LLM Routing code document to quickly conduct ablation experiments. In the following example, Feature extraction represents the training of data extracted from semantic and syntactic features, and original represents the training of original data.

```python
python ablation_ features.py 
--raw_data ./data/original 
-- Feature extraction _data ./data/ Feature extraction
```

Table 5

| Model                                        | CPT（50%） | CPT（80%） | APGR  |
| -------------------------------------------- | ---------- | ---------- | ----- |
| No semantic and syntactic feature extraction | 38.75%     | 72.65%     | 0.568 |
| semantic and syntactic feature extraction    | 37.25%     | 71.89%     | 0.589 |



## Comparison of experiments with and without building a RAG framework

You can create scripts main_rag_ablation.py and main_unrag_ablation.py based on the uploaded LLM Routing code document to quickly conduct ablation experiments. In the following example, main_rag_ablation.py represents training by using RAG to build a vector database and integrating the K-nearest neighbor algorithm for similarity retrieval,while main_unrag_ablation.py represents training without building a RAG vector database.



Enable the RAG framework to run the example:

```python
python main_rag_ablation.py

--dpath ./data/test data MMLU 

--arch QWEN-7B 

--rag_mode 

--k_neighbors 3
```

Disable the RAG framework to run the example

```
Python main_unrag_ablation.py 

--test dataset MMLU

--arch QWEN-7B 
```

Table 6

| Model                                | CPT（50%） | CPT（80%） | APGR  |
| ------------------------------------ | ---------- | ---------- | ----- |
| No RAG vector knowledge base         | 39.68%     | 73.78%     | 0.559 |
| Building a RAG vector knowledge base | 37.25%     | 71.89%     | 0.589 |

