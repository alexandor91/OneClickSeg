"""
注意力机制的计算, 以及封装成 attention layer/block

默认 batch_first=True
仿照 nn.MultiheadAttention 和 F.multi_head_attention_forward, 只写 attn_drop;
layer norm、attention和ffn linear/activation 后的drop out、shortcut、positional embedding, 都写在 transformer layer里;
注意在 MHA 中实现 attn_mask、key_padding_mask
"""

import torch
import torch.nn as nn

# from torch.nn.init import xavier_uniform_, constant_, xavier_normal_


# -------------------------------------------------------------------------
# 普通的 attn, from https://github.com/sunjiahao1999/SPFormer
# 参考了 Mask3D, 但改成了 squeezed batch 版本: 对不定长 q k, 只能在自己 batch 内查询, 不然attn会互相影响, 因为做了softmax
# 点云由于不同批次点数不同, 如果 padding 到相同长度, 容易爆显存
# 能用 attn_mask 实现 batched attention 吗？不能, 因为 attn_mask 无法避免跨 batch 的归一化
# -------------------------------------------------------------------------
class SqueezedSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, layer_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(layer_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_batch, pe=None):
        """
        - x: Tensor (n_x, embed_dim)
        - x_batch: (n_x), which batch x belongs to
        - pe: Tensor (n_x, embed_dim)
        - output: Tensor (n_x, embed_dim)
        """
        bs = x_batch.max() + 1
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)
        x_with_pe = self.with_pos_embed(x, pe)
        for b in range(bs):
            batch_mask = x_batch == b
            q = k = x_with_pe[batch_mask]
            v = x[batch_mask]
            attn_out, _ = self.attn(q, k, v)
            attn_out = self.dropout(attn_out) + v
            attn_out = self.norm(attn_out)
            output[batch_mask] = attn_out
        return output


class SqueezedCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, layer_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(layer_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        source,
        query,
        source_batch,
        query_batch,
        attn_mask=None,
        source_pe=None,
        query_pe=None,
    ):
        """
        - source: Tensor (n_s, embed_dim)
        - source_batch: (n_s), which batch the source belongs to
        - query: Tensor (n_q, embed_dim)
        - query_batch: (n_q), which batch the query belongs to
        - attn_mask: (n_q, n_s)
        - source_pe: (n_s, embed_dim)
        - query_pe: (n_q, embed_dim)
        - output: (n_q, embed_dim)
        """
        assert (
            query_batch.max() <= source_batch.max()
        ), "err: query_batch.max() > source_batch.max()"
        bs = query_batch.max() + 1
        output = torch.empty_like(query, dtype=query.dtype, device=query.device)
        source_with_pe = self.with_pos_embed(source, source_pe)
        query_with_pe = self.with_pos_embed(query, query_pe)
        # 遍历 query 的每个点太慢, 因此遍历每个batch
        for b in range(bs):
            # b = query_batch[i]
            source_batch_mask = source_batch == b
            query_batch_mask = query_batch == b
            # q = query_with_pe[i]         # (3, embed_dim)
            q = query_with_pe[query_batch_mask]  # (n_q_b, embed_dim)
            k = source_with_pe[source_batch_mask]  # (n_s_b, embed_dim)
            v = source[source_batch_mask]
            if attn_mask is not None:
                attn_mask_b = attn_mask[query_batch_mask]  # (n_q_b, n_s)
                attn_mask_b = attn_mask_b[:, source_batch_mask]  # (n_q_b, n_s_b)
                # 数值稳定性: 如果 attn_mask 全为 True, 则不 mask 掉, 确保一个 query 至少能查到一个点
                attn_mask_b[
                    torch.where(attn_mask_b.sum(-1) == attn_mask_b.shape[-1])
                ] = False
                attn_out, _ = self.attn(
                    q, k, v, attn_mask=attn_mask_b
                )  # (n_q_b, embed_dim)
            else:
                attn_out, _ = self.attn(q, k, v)
            # attn_out = self.dropout(attn_out) + query[i]      # (3, embed_dim)
            attn_out = self.dropout(attn_out) + query[query_batch_mask]
            attn_out = self.norm(attn_out)
            output[query_batch_mask] = attn_out.to(query.dtype)
        return output


# -------------------------------------------------------------------------
# 批量计算 self attn, 速度快, 但显存占用大
# -------------------------------------------------------------------------
class BatchedSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, layer_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(layer_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        - x: Tensor (b, n_x, embed_dim)
        - pe: Tensor (b, n_x, embed_dim)
        - output: Tensor (b, n_x, embed_dim)
        """
        x_with_pe = self.with_pos_embed(x, pe)
        q = k = x_with_pe
        v = x
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.dropout(attn_out) + v
        attn_out = self.norm(attn_out)
        return attn_out
