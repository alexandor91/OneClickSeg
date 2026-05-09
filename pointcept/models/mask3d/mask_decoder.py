"""
参考的 Mask2Former, Mask3D 的设计

transformer layer, transformer decoder, mask decoder
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import spconv.pytorch as spconv

import warnings
from typing import Tuple, Type, List, Dict
from collections.abc import Sequence

from .misc import FFNLayer, GenericMLP
from .attention import (
    SqueezedCrossAttentionLayer,
    SqueezedSelfAttentionLayer,
    BatchedSelfAttentionLayer,
)

from pointcept.utils_iseg.structure import Query, Scene
from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule, PointSequential


# ------------------------------------------------------------------------
# 参考 transformer.py, agile3d 的 decoder layer
# ------------------------------------------------------------------------
@MODELS.register_module()
class Mask3dDecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 1024,
        activation: str = "relu",
        attn_drop: float = 0.0,
        layer_drop: float = 0.0,
        norm_before=False,
    ) -> None:
        """
        A transformer block with three layers:
        1. masked query to source attention, using instance pred before to refine as mask
        2. query to query attention
        3. ffn layer

        Arguments:
        - embedding_dim (int): the channel dimension of the embeddings, same as click embedding_dim
        - num_heads (int): the number of heads in the attention layers
        - mlp_dim (int): the hidden dimension of the mlp block
        - activation (str): the activation name of the mlp block
        - attn_drop (float): dropout percent in each attn layer, default 0.
        - layer_drop (float): dropout percent in each dropout layer, default 0.
        - norm_before (bool): whether to normalize before attn (unused)
        """
        super().__init__()

        self.masked_query_to_source = SqueezedCrossAttentionLayer(
            embedding_dim, num_heads, attn_drop=attn_drop, layer_drop=layer_drop
        )

        self.query_to_query = BatchedSelfAttentionLayer(
            embedding_dim, num_heads, attn_drop=attn_drop, layer_drop=layer_drop
        )

        self.mlp = FFNLayer(
            embedding_dim,
            mlp_dim=mlp_dim,
            activation=activation,
            dropout=layer_drop,
            normalize_before=norm_before,
        )

    def forward(
        self,
        source: Tensor,
        query: Tensor,
        source_batch: Tensor,
        query_batch: Tensor,
        source_pe=None,
        query_pe=None,
        attn_mask=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
        - source (torch.Tensor): source pcd embedding. Shape as (N_source, embedding_dim).
        - query (torch.Tensor): query embedding. Shape as (N_query, embedding_dim).
        - source_batch (torch.Tensor): shape as (N_source,).
        - query_batch (torch.Tensor): shape as (N_query,).
        - source_pe (torch.Tensor): the positional encoding to add to the source. Must have the same shape as source.
        - query_pe (torch.Tensor): the positional encoding to add to the query. Must have the same shape as query.
        - attn_mask (torch.Tensor, default None): mask for masked_query_to_source layer. Shape as (N_query, N_source).

        Returns:
        - torch.Tensor: the processed source embedding
        - torch.Tensor: the processed query embedding
        """
        query_after_cross = self.masked_query_to_source(
            source=source,
            query=query,
            source_batch=source_batch,
            query_batch=query_batch,
            attn_mask=attn_mask,
            source_pe=source_pe,
            query_pe=query_pe,
        )
        query_after_self = self.query_to_query(x=query_after_cross, pe=query_pe)
        query_final = self.mlp(query_after_self)
        return source, query_final


@MODELS.register_module()
class Mask3dMaskDecoder(PointModule):
    def __init__(
        self,
        transformer_block_cfg,
        num_classes: int,
        attn_mask_types: Sequence,
        enable_final_block=True,
        num_decoders=3,
        shared_decoder=True,
        mask_num=1,
    ) -> None:
        """
        用transformer、head 计算 mask
        Args:
        - num_classes: 语义labels的类别数
        - attn_mask_types: 一个 decoder 的每层 transformer block 的 attn_mask 类型
            - str, ['none', 'bool', 'float']
        - enable_final_block: 是否对最顶层 feature map 也用 transformer block 去 query
        - num_decoders: decoder 堆叠数量, 默认 3, 每一个都是完整的多分辨率 transformer decoder
        - shared_decoder: 堆叠的 decoder 是否共享参数, 默认 True
        - mask_num: TODO 增加多尺度mask

        Return: mask_logits, cls_logits
        - mask_logits: 场景中点属于哪个click对应的mask, N_query x N_point[-1] (* mask_num)
        - cls_logits: click点对应的类别, N_query x num_classes
        """
        super().__init__()
        self.embedding_dim = transformer_block_cfg.embedding_dim
        self.attn_mask_types = attn_mask_types
        self.enable_final_block = enable_final_block
        self.features_num = len(attn_mask_types)
        self.num_classes = num_classes

        # transformer blocks
        self.num_decoders = num_decoders
        self.shared_decoder = shared_decoder
        n = num_decoders + shared_decoder * (
            1 - num_decoders
        )  # 共享参数, 则只设置一个 decoder
        self.blocks = nn.ModuleList([nn.ModuleList() for _ in range(n)])
        for i in range(n):
            for j in range(self.features_num - 1 + self.enable_final_block):
                self.blocks[i].append(MODELS.build(transformer_block_cfg))

        # head
        # TODO 统一线性层, MLP和FFN
        self.norm_before_head = nn.LayerNorm(self.embedding_dim)
        self.mask_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.cls_head = nn.Linear(self.embedding_dim, self.num_classes)

    def downsample(self, x, scene: Scene, from_resolution=0, to_resolution=0):
        """
        将 x 的 dim 0, 按照 indices, 从 from_resolution 分辨率, 下采样到 to_resolution 分辨率; x 为 None 时返回 None
        """
        if x is None:
            return None
        return scene.feature_pyramid.downsample(x, from_resolution, to_resolution)

    @torch.no_grad()
    def compute_attn_mask(self, masks_logits: Tensor, scene, j, tau=0.5):
        """构建第 j 个 decoder stage 的 attn_mask
        - masks_logits: N_query x N_point[-1]
        - scene: scene data dict, 为了降采样
        - j: 当前 decoder stage
        - tau: 阈值, mask 掉小于 tau 的值\n
        Return:
        - attn_mask: N_query x N_point[j]
        """
        if self.attn_mask_types[j] == "none":
            return None
        attn_mask = masks_logits.clone().detach()  # (N_query, N_point[-1])
        attn_mask = self.downsample(
            attn_mask.T.contiguous(),
            scene,
            from_resolution=self.features_num - 1,
            to_resolution=j,
        ).T.contiguous()  # (N_query, N_point[-1]) -> (N_query, N_point[j])
        # float type, add to attn weight
        # bool type, positions with ``True`` are not allowed to attend
        # while ``False`` values will be unchanged.
        if self.attn_mask_types[j] == "bool":
            attn_mask = attn_mask.sigmoid() < tau
        # 注意在 masked attention 中, 不要让一次 attn 操作全部被mask掉, 会返回 nan
        return attn_mask

    def head(self, scene_feat, query_feat):
        """head
        - scene_feat: N_point[-1] x embedding_dim
        - query_feat: N_query x embedding_dim\n
        Return:
        - masks_logits: N_query x N_point[-1]
        - cls_logits: N_query x num_classes
        """
        query_feat = self.norm_before_head(query_feat)
        # 只在 loss 中计算 masks_heatmap
        # 因为在混合精度训练 amp.autocast(True) 时, .sigmoid() 容易有数值稳定性问题
        # TODO 要不要禁止跨batch匹配, 强制query和source在同一个 batch 内?
        mask_feat = self.mask_head(query_feat)  # N_query x embedding_dim
        masks_logits = torch.einsum(
            "q d, p d -> q p", mask_feat, scene_feat
        )  # N_query x N_point
        cls_logits = self.cls_head(query_feat)  # N_query x num_classes
        return masks_logits, cls_logits

    def transformer_fwd(self, scene: Scene, query: Query, pred_init: dict):
        """
        transformer forward
        - scene: N_point data dict
        - query: query data dict
        - pred_init: initial pred dict, 用于生成 initial attn mask
            - masks_logits: N_query x N_point[-1]
            - cls_logits: N_query x num_classes\n
        Return:
        - preds: list(dict), 包含 num_decoders x (features_num - 1 + self.enable_final_block) 个 decoder 的输出
            - 'masks_logits': N_query x N_point[j // num_decoders]
            - 'cls_logits': N_query x num_classes
        """
        # aux loss, 中间层也可用于监督, 包括最开始用于生成 initial mask 的 head
        masks_logits = pred_init["masks_logits"]
        preds = [pred_init]
        for i in range(self.num_decoders):
            # 堆叠 num_decoders 个 decoder
            i = i * (1 - self.shared_decoder)  # 共享参数, i 置 0
            for j in range(self.features_num - 1 + self.enable_final_block):
                # (features_num - 1 + self.enable_final_block) 个 block
                attn_mask = self.compute_attn_mask(masks_logits, scene, j)
                scene_feat = scene.feat_list[j]  # (N_point[j], embedding_dim)
                # 需要对 feat 进行排序吗? feat and scene_pe must be corresponding in point-wise
                # 不需要, 因为经过同样的 sparse downsample 过程, 结果的顺序也一致
                scene_feat, query.feat = self.blocks[i][j](
                    source=scene_feat,
                    query=query.feat,
                    source_pe=scene.pe_list[j],
                    query_pe=query.pe,
                    source_batch=scene.batch_list[j],
                    query_batch=query.batch,
                    attn_mask=attn_mask,
                )
                masks_logits, cls_logits = self.head(scene.mask_feat, query.feat)
                preds.append({"masks_logits": masks_logits, "cls_logits": cls_logits})
        return preds

    def forward(self, scene: Scene, query: Query, pred_last):
        """
        - scene: N_point data dict
            - feature_pyramid: features collection layer
            - feat_list: list, feat_list[j] == (N_point[j], embedding_dim)
            - pe_list: list, pe_list[j] == (N_point[j], embedding_dim)
            - batch_list: list, batch_list[j] == (N_point[j],), 一般可以直接从 scene.feature_pyramid.features[j] 中获取
        - query: query data dict
            - feat: N_query x embedding_dim
            - pe: positional encoding, N_query x embedding_dim
            - batch: (N_query,)
        - pred_last: dict, 上一次的预测结果
            - 'masks_logits': N_query_last x N_point[-1], corresponding to query.feat and query.pe
            - 'cls_logits': N_query_last x num_classes
        Return:
        - preds: list(dict), 包含 num_decoders x (features_num - 1 + self.enable_final_block) 个 decoder 的输出
            - 'masks_logits': N_query x N_point[j // num_decoders]
            - 'cls_logits': N_query x num_classes
        """
        # 新query, 利用scene embedding生成pred_logits, N_query_last -> N_query
        last_masks_logits = pred_last["masks_logits"].clone().detach()
        last_cls_logits = pred_last["cls_logits"].clone().detach()
        N_query, N_query_last = (
            query.feat.shape[0],
            last_masks_logits.shape[0],
        )  # 默认 query 中新的靠后
        if N_query > N_query_last:
            N_query_new = N_query - N_query_last
            masks_padding, cls_padding = self.head(
                scene.mask_feat, query.feat[-N_query_new:]
            )
            masks_logits = torch.cat(
                [last_masks_logits, masks_padding], dim=0
            )  # (N_query, N_point[-1])
            cls_logits = torch.cat(
                [last_cls_logits, cls_padding], dim=0
            )  # (N_query, num_classes)
        elif N_query == N_query_last:
            masks_logits = last_masks_logits
            cls_logits = last_cls_logits
        else:
            raise ValueError(f"assert {N_query} >= {N_query_last}")

        pred_init = {
            "masks_logits": masks_logits.contiguous(),
            "cls_logits": cls_logits.contiguous(),
        }
        preds = self.transformer_fwd(scene, query, pred_init)
        return preds
