import torch
from numpy import dtype
from numpy.f2py.auxfuncs import throw_error
from torch import nn, Tensor
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Qwen2Config, PreTrainedTokenizer, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Attention, logger, Qwen2RotaryEmbedding, \
    apply_rotary_pos_emb, repeat_kv, Cache, Qwen2DecoderLayer, Qwen2Model, QWEN2_INPUTS_DOCSTRING, Qwen2RMSNorm
from typing import List, Optional, Tuple, Union, Any
from torch import nn
from transformers.utils import add_start_docstrings_to_model_forward


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class KGQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, embed_tokens, weight_k, weight_q):
        super().__init__(config, layer_idx)
        self.encode_relation = KGEmbedding(self.hidden_size, self.hidden_size, 2, config.vocab_size, weight_q, weight_k, config.num_attention_heads)
        self.embed_tokens = embed_tokens

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
        ############################加入知识图谱###########################

        hidden_states = self.encode_relation(hidden_states, attention_mask, self.embed_tokens)

        ############################加入知识图谱###########################

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

class KGQwen2Model(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化参数
        glorot(self.W_k.weight)
        glorot(self.W_q.weight)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_q.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        inputs_mask = attention_mask
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KGQwen2Attention(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None, relation_model: nn.Module = None):
        super().__init__(config, layer_idx)
        self.encode_relation = KGEmbedding(self.hidden_size, self.hidden_size, 2, config.vocab_size)

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


class KGQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.token_dim = self.model.embed_tokens.embedding_dim
        self.encode_relation = KGEmbedding(self.token_dim, self.token_dim, 2, config.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tokenizer : Optional[PreTrainedTokenizer] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        #########################################################################
        # 依据输出的hidden_states和self.encode_relation增强hidden_states

        hidden_states = self.encode_relation(hidden_states, input_ids, attention_mask, self.model.embed_tokens)

        #########################################################################
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def rest_parameter(value):
    if isinstance(value, torch.Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)


def extract_valid_token_mask(attention_mask):
    """
    将各种形状的 attention_mask 转换成 (batch, seq_len) 的有效 token mask
    """
    if attention_mask.dim() == 2:
        # (batch, seq_len)
        return attention_mask.bool()

    elif attention_mask.dim() == 3:
        # (batch, 1, seq_len) -> squeeze
        return attention_mask.squeeze(1).bool()

    elif attention_mask.dim() == 4:
        # (batch, 1, seq_len, seq_len)
        # 提取对角线，表示 token 自己是否有效
        return (attention_mask[:, 0] > -1e4).any(dim=-1).bool()

    else:
        raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")


class KGEmbedding(nn.Module):
    #  需从新设计关系编码器 由W^{N \times N \times R}、W_{q}^{N \times N}、W_{k}^{N \times N}、W_{ATT}^{N \times N}构成
    def __init__(self, channels, hidden_channels, relation_num, node_num, weight_q, weight_k, num_heads):
        super(KGEmbedding, self).__init__()
        self.k = 4096
        self.channels = channels
        self.relation_num = relation_num
        self.hidden_channels = hidden_channels
        # 通过W_q、W_k去计算X_i与X_j之间的相关性而不是显示存储在R_map中
        self.num_heads = num_heads
        self.head_dim = self.hidden_channels // self.num_heads
        self.W_q = weight_q
        self.W_k = weight_k
        self.W_v = nn.Linear(channels, channels)
        self.update = nn.Linear(channels, channels)
        # self.R_i = torch.arange(node_num, dtype=torch.int)  # 映射不同节点关系索引
        # 初始化参数
        self._init_weights()
        self.alpha = 0.2

    def _init_weights(self):
        # 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.update.weight)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.update.bias)

    def _encode_relation(self, h_t, attention_mask, embedding: nn.Embedding):
        bsz, node_num, channels = h_t.size()
        device = h_t.device
        dtype = h_t.dtype
        x_residual = h_t

        if attention_mask is not None:
            attention_mask = extract_valid_token_mask(attention_mask)
            mask = attention_mask.bool()
            assert mask.sum() != 0, "attention_mask掩码为空"
            h_t = self.aggregate_n(h_t, mask, embedding, True)  # [B, V_s, V_t]
            h_t = x_residual + self.update(h_t)

        else:
            throw_error("attention mask is required.")

        return h_t

    def aggregate_n(self, h_t, attention_mask, embedding, keepdim=True):
        # 保持输入输出 dtype 一致
        input_dtype = h_t.dtype

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            # 直接取 embedding.weight
            token_embedding = embedding.weight  # [V, d]

            # batch 有效 token
            bsz = attention_mask.sum()
            nodes = token_embedding.size(0)

            # 确保使用与输入相同的精度
            dtype = h_t.dtype

            # Q/K/V 投影
            x_target = self.W_k(token_embedding).view(self.num_heads, -1, self.head_dim)  # [V, d]
            x_start = self.W_q(h_t[attention_mask, :]).view(self.num_heads, -1, self.head_dim)  # [B, d]
            x_v = self.W_v(token_embedding).view(self.num_heads, -1, self.head_dim)

            # with torch.cuda.amp.autocast():  # AMP 节省显存
            score_s2t = torch.matmul(x_start, x_target.transpose(1, 2)) / math.sqrt(self.head_dim)  # [V, s]

            target = torch.zeros(bsz, self.num_heads, nodes, dtype=torch.bool, device="cuda")  # [bsz, nodes]

            # 使用dropout代替随机采样
            score_s2t = torch.softmax(score_s2t, dim=-1, dtype=torch.float32).to(input_dtype)
            score_s2t = score_s2t.reshape(-1, self.num_heads, nodes)
            score_s2t = F.dropout(score_s2t, p=0.25, training=self.training)
            topk_val, topk_idx = torch.topk(score_s2t, k=min(self.k, nodes), dim=-1)
            target[:] = False
            target.scatter_(2, topk_idx, True)
            score_s2t = score_s2t.masked_fill(~target, float(0.0))  # 只保留mask的元素

            h_hat_t = torch.einsum('bhd,hdk->bhk', score_s2t, x_v)
            h_hat_t = h_hat_t.reshape(-1, self.hidden_channels)

        h_t_new = h_t.clone()
        h_t_new = h_t_new.to(input_dtype)
        h_t_new[attention_mask] = h_hat_t
        return h_t_new

    def update_n(self, h_t):
        return F.gelu(self.update(h_t))

    def _forward(self, query_states, attention_mask, embedding: nn.Embedding):
        return self._encode_relation(query_states, attention_mask, embedding)

    def forward(self, query_states, attention_mask, embedding: nn.Embedding):
        return self._forward(query_states, attention_mask, embedding)

