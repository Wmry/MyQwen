# DeepSeek结构学习
## 模型结构
```python
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```
### Embedding层
嵌入层通过输入的词索引给出相应的词嵌入向量Embedding(字典中词的数量，嵌入维度)

# 评价指标

```bash
pip install evaluate # 使用hugging face的评价指标库 evaluate
```

# 目标函数

## 任务
### 设置专业知识语料库交由DeepSeek给出训练集和测试集



### 设置知识图谱在进行生成模型推理时汇入词向量的相关系数

#### 相关论文
[(2020) K-ADAPTER: Infusing Knowledge into Pre-Trained Models with Adapters](D:/Users/xiangyu/download/paper/K-ADAPTER.pdf)
将知识三元组注入Transformer输入序列，设计可见性矩阵控制知识传播

##### GNN-LM 

[(2022)GNN-LM: LANGUAGE MODELING BASED ON GLOBAL CONTEXTS VIA GNN](D:/Users/xiangyu/download/paper/GNN-LM.pdf)
使用GNN构建知识图谱并更新大模型知识，Understand_step_1:使用$V \in $


#######  构建知识图谱
<font color=blue size=3>设计思路：通过设计编码器（线性层）构建隐向量通过内积计算相似度，代表不同节点之间的关系，不再明确存储节点之间的关系。</font>
<font color=red size=3>问题1：如何使得隐空间有区分</font>

[(2025)AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding](D:/Users/xiangyu/download/paper/AlignVLM.pdf)
关键技术：对比学习实现多模态对齐
[(2025)Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](D:/Users/xiangyu/download/paper/Suervey on agentix RAG.pdf)
RAG综述
[(2025)MEFT:Memory-Efficient Fine-Tuning through Sparse Adapter](D:/Users/xiangyu/download/paper/MEFT.pdf)
关键技术：轻量级知识编辑模块



[TZ](https://lei-su.com/#/dashboard)