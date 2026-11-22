"""Batched Qwen attention for keyframes on the same canvas."""

import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision


def vision_attention(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    images = cu_seqlens.numel() - 1
    query_states, key_states, value_states = (
        tensor.reshape(images, -1, self.num_heads, self.head_dim).transpose(1, 2)
        for tensor in (query_states, key_states, value_states))
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, dropout_p=0.0, scale=self.scaling)
    attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)
