# DeepSeek结构学习
<font color=red>红色表示暂定需要确定的</font>

<font color=blue>蓝色表示可暂定确定的</font>

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

######  构建知识图谱 ( 如何构建上下文异构图 ) 
<font color=blue size=3>设计思路1、**在经过LM形成最终隐向量后**，通过设计编码器（隐状态+线性层）构建隐向量通过内积计算相似度，代表不同节点之间的关系，不再明确存储节点之间的关系生成隐向量汇聚的节点为embed_tokens中获取的隐状态，边通过余弦相关性计算，通过Droput和TopK(4096)训练节点，学习语料中的相关性</font><font color=red size=2>（在每层Attention生成语句后进行融合）。</font> GraphSAGE可不使用原GNN的汇聚范式，而是使用Transformer的方式去汇聚节点信息，使用mask掩码去调节汇聚的邻居节点、k阶邻近就是进行多少次Transformer，最后得到有上下文信息的Token再跟输入的Token进行汇聚，类似于Graphormer 方法，但是在构建mask时考虑GraphSAID方法，通过将大图（超过1万节点）分解为子图进行逐步汇聚
$$
通过构建可学习的语料关系编码器，存储语料中不同Token之间的关系（类似于连续上下文），最终构建一个W\in R^{N\times N \times R}\\
同时通过W_{k}^{C\times C},W_{q}^{C \times C}，W_{ATT}^{C\times C}辅助映射三元组\\
h_t
$$
$$
N(c_t) = \{c^{(1)}_{t_1}, ..., c^{(k)}_{t_k}\}\\
p=(W_t|C_t)\\
c^{(i)}_j = \{w^{(i)}_{j+p}\}^r_{p=-l}，l,r表示左右窗口大小\\
w^{(i)}_j\\i表示i^{th}训练样本，j表示j^{th}时间步，w^{(i)}_j,通过查询模型nn.Embedding获取\\
x_s^{n_s\times C}W_q^{C \times C}W_k^{C \times C}x_t^{C\times n_t} = ATT^{n_S\times n_t}\\
可进一步简化为W_q^{C \times C}= W + \delta_q, W_k^{C \times C}=W + \delta_k 方法如下代码
$$
```python
import torch
import torch.nn as nn

class SharedQK(nn.Module):
    def __init__(self, dim, rank=16):
        super().__init__()
        # 主干矩阵 (共享)
        self.W = nn.Parameter(torch.randn(dim, dim))
        # 修正项 (低秩或小参数)
        self.delta_q = nn.Parameter(torch.randn(dim, rank) @ torch.randn(rank, dim))
        self.delta_k = nn.Parameter(torch.randn(dim, rank) @ torch.randn(rank, dim))

    def forward(self, x_s, x_t):
        W_q = self.W + self.delta_q
        W_k = self.W + self.delta_k

        Q = x_s @ W_q
        K = x_t @ W_k
        attn = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        return attn

```
\*预训练方案：通过逻辑掩码+回归预测任务训练关系编码器。通过scatter_add进行汇聚，原论文中的$h_s, h_n$通过预训练存储再通过【FAISS】存储预料中Token向量并建立索引

![sctter_add操作示意图](C:\Users\LZF\Desktop\sctter_add操作示意图.jpg)

<font color=blue size=3>设计思路2、**在经过LM形成最终隐向量后**，通过设计编码器（隐状态+激活函数（GELU）+线性层）构建隐向量通过内积计算相似度，代表不同节点之间的关系，不再明确存储节点之间的关系</font><font color=red size=2>（在每层Attention生成语句后进行融合）。</font>

<font color=red size=3>实验记录（2-25-09-15）、**TopK 256训练效果太差**，需验证是训练参数有问题、还是模型具体实现有问题</font><font color=blue size=2>（需查看论文分析，为何验证时准确率为0）。</font>

[(2025)AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding](D:/Users/xiangyu/download/paper/AlignVLM.pdf)
关键技术：对比学习实现多模态对齐
[(2025)Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](D:/Users/xiangyu/download/paper/Suervey on agentix RAG.pdf)
RAG综述
[(2025)MEFT:Memory-Efficient Fine-Tuning through Sparse Adapter](D:/Users/xiangyu/download/paper/MEFT.pdf)
关键技术：轻量级知识编辑模块



[TZ](https://lei-su.com/#/dashboard)