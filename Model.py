import torch
from torch import nn
import math
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Attention, logger, Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv, Cache, Qwen2DecoderLayer
from typing import List, Optional, Tuple, Union
from torch import nn


class KGQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states # 获取残差向量

        hidden_states = self.input_layernorm(hidden_states) # 正则化

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states # 短残差链接

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class KGQwen2Attention(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None, relation_model: nn.Module = None):
        super().__init__(config, layer_idx)
        self.relation_model = relation_model

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        ################################################################################################################
        # 嵌入对模型改造的相关代码 计算未加入词向量的关系矩阵
        # self.relation_model(query_states, key_states)

        ################################################################################################################

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        ################################################################################################################
        # 在此处插入计算knowledge Graph Relation的词嵌入kg_relation_output,通过权重W将其融入attn_output中
        # torch.matmul(attn_weights, value_states)  attn_weights size (bsz, head_attn, q_len, q_len)
        #                                           value_states size (bsz, head_attn, q_len, channels)

        ################################################################################################################

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class KGEmbedding(nn.Module):
    #  需从新设计关系编码器 由W^{N \times N \times R}、W_{q}^{N \times N}、W_{k}^{N \times N}、W_{ATT}^{N \times N}构成
    def __init__(self, channels, hidden_channels, relation_num, node_num):
        super(KGEmbedding, self).__init__()
        self.channels = channels
        self.relation_num = relation_num
        self.hidden_channels = hidden_channels
        self.W_q = nn.Linear(channels, channels)
        self.W_k = nn.Linear(channels, channels)
        self.R_map = nn.Parameter(torch.Tensor(node_num, node_num, relation_num))

    def _encode_relation(self, h_t, layer_id, embedding: nn.Embedding):
        (bsz, node_num, channels) = h_t.size()
        layer_id = self.R_map[layer_id, :, :] # (bsz, 1, :, relation_num)
        node_e = embedding(layer_id) # (bsz, :, relation_num, channels)
        h_t_e = self.W_q(node_e) # (bsz, :, channels)
        h_t_s = self.W_k(h_t) # (bsz, node_num, channels)
        attention = (torch.matmul(h_t_s, h_t_e.transpose(2, 3)) * self.R_map[layer_id, :, :]) / math.sqrt(self.channels)
        return attention

    def _forward(self, query_states, key_states):
        input_dtype = query_states.dtype
        (bsz, n_len, channels) = query_states.shape
        x1 = self.encode_relation(torch.cat([query_states, key_states], dim=-1))
        x1 = x1.reshape(bsz*self.relation_num, n_len, -1)
        # upcast attention to fp32
        x1 = x1.to(torch.float32)
        x2 = torch.bmm(x1, x1.transpose(1, 2))
        relation = x2.reshape(bsz, -1, n_len, n_len)
        return relation.to(input_dtype)

    def forward(self, query_states, key_states):
        return self._forward(query_states, key_states)
