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
        norm_first=False,
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
        - norm_first (bool): whether to normalize before attn (unused)
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
            normalize_before=norm_first,
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
        query = self.masked_query_to_source(
            source=source,
            query=query,
            source_batch=source_batch,
            query_batch=query_batch,
            attn_mask=attn_mask,
            source_pe=source_pe,
            query_pe=query_pe,
        )
        query = self.query_to_query(x=query, pe=query_pe)
        query = self.mlp(query)
        return source, query


@MODELS.register_module()
class Mask3dMaskDecoder(PointModule):
    def __init__(
        self,
        transformer_block_cfg,
        features_dims: Sequence,
        query_embedding_dim: int,
        mask_head_hidden_dims: Sequence,
        cls_head_hidden_dims: Sequence,
        num_classes: int,
        with_attn_mask: Sequence,
        enable_final_block=False,
        mask_num=1,
    ) -> None:
        """
        用transformer、head 计算 mask
        TODO scene 和 query 的构建放到顶层
        Args:
        - features_dims: 输入的不同分辨率的 features 各自的 embedding_dim 大小,
            - sorted from coarse to fine, len(features_dims) == features_num
            - 有几张feature map就用几层 transformer block; in_proj_layers 用到
        - query_embedding_dim: int, in_proj_query 用
        - num_classes: 语义labels的类别数
        - with_attn_mask: 每层 transformer block 是否需要 attn_mask, same len as features_dims
        - enable_final_block: 是否对最顶层 feature map 也用 transformer block 去 query
        - mask_num: TODO 增加多尺度mask

        Return: mask_logits, cls_logits
        - mask_logits: 场景中点属于哪个click对应的mask, N_point[-1] x N_query (* mask_num)
        - cls_logits: click点对应的类别, N_query x num_classes
        """
        super().__init__()
        self.features_dims = features_dims
        self.features_num = len(features_dims)
        self.embedding_dim = transformer_block_cfg.embedding_dim
        self.with_attn_mask = with_attn_mask
        self.enable_final_block = enable_final_block
        assert (
            len(with_attn_mask) == self.features_num
        ), f"len(with_attn_mask) must be same as feature_maps_dims {self.features_num}, but got{len(with_attn_mask)}"
        self.num_classes = num_classes

        # in_proj && transformer blocks
        self.in_proj_query = nn.Linear(query_embedding_dim, self.embedding_dim)
        self.in_proj_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.features_num):
            in_proj = nn.Linear(features_dims[i], self.embedding_dim)
            self.in_proj_layers.append(in_proj)
            self.blocks.append(
                nn.Identity()
                if i == self.features_num - 1 and not enable_final_block
                else MODELS.build(transformer_block_cfg)
            )

        # head
        self.norm_before_head = nn.LayerNorm(self.embedding_dim)
        self.mask_head_mlp = GenericMLP(
            input_dim=self.embedding_dim,
            hidden_dims=mask_head_hidden_dims,
            output_dim=self.embedding_dim,
            norm_fn_name=None,
            activation="relu",
            use_conv=False,
            dropout=0.0,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform",
        )
        self.cls_head_mlp = GenericMLP(
            input_dim=self.embedding_dim,
            hidden_dims=cls_head_hidden_dims,
            output_dim=self.embedding_dim,
            norm_fn_name=None,
            activation="relu",
            use_conv=False,
            dropout=0.0,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform",
        )
        self.sem_proto = nn.Embedding(num_classes, self.embedding_dim)
        self.query_proj = nn.Linear(3, 1)
        self.sdf_head = GenericMLP(
            input_dim=self.embedding_dim + 3,
            hidden_dims=mask_head_hidden_dims,
            output_dim=1,
            norm_fn_name=None,
            activation="relu",
            use_conv=False,
            dropout=0.0,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform",
        )
        self.proto_head = GenericMLP(
            input_dim=self.embedding_dim * 2,
            hidden_dims=mask_head_hidden_dims,
            output_dim=self.embedding_dim,
            norm_fn_name=None,
            activation="relu",
            use_conv=False,
            dropout=0.0,
            hidden_use_bias=True,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False,
            weight_init_name="xavier_uniform",
        )

    def in_projection(self, features):
        """
        - features: list, features[j] == (N_point[j], self.features_dims[j])
        - features_out: list, features_out[j] == (N_point[j], embedding_dim)
        """
        features_out = []
        for j in range(self.features_num):
            feat = features[j]  # (N_point[j], self.features_dims[j])
            feat = self.in_proj_layers[j](feat.float())  # (N_point[j], embedding_dim)
            features_out.append(feat)
        return features_out

    def downsample(self, x, scene: Scene, from_resolution=0, to_resolution=0):
        """
        将 x 的 dim 0, 按照 indices, 从 from_resolution 分辨率, 下采样到 to_resolution 分辨率; x 为 None 时返回 None
        """
        if x is None:
            return None
        return scene.feature_pyramid.downsample(x, from_resolution, to_resolution)

    def mask_module(self, scene_feat, query_feat, scene, j, tau=0.5):
        """seg head 和构建 attn_mask, 层之间的 attn_mask 需要 detach
        - scene_feat: N_point[-1] x embedding_dim
        - query_feat: N_query x embedding_dim
        - scene: scene data dict, 为了降采样
        - j: 当前阶段, 需要降采样到 j+1 阶段
        - tau: 阈值, attn mask 去掉小于 tau 的值
        TODO 加上那个向外扩展的 query
        """
        query_feat = self.norm_before_head(query_feat)
        cls_logits = self.cls_head(query_feat)
        masks_heatmap = self.mask_head(scene_feat, query_feat, scene)
        # 1. new mask query, merge its corresponding scene feat
        mask_query = masks_heatmap @ scene_feat.T  # N_query x embedding_dim
        mask_query = self.norm_before_head(mask_query)
        # 2. new cls query, select its corresponding aligned proto
        cls_query = self.align_proto[cls_logits.argmax(-1)]  # N_query x embedding_dim
        cls_query = self.norm_before_head(cls_query)
        # 3. attn mask, If a BoolTensor, ``True`` means not allowed to attend
        attn_mask = masks_heatmap.clone().detach() < 1 - tau
        with torch.no_grad():
            attn_mask_down = self.downsample(
                attn_mask,
                scene,
                from_resolution=self.features_num - 1,
                to_resolution=j + 1,
            ).T  # (N_point[-1], N_query) -> (N_query, N_point[j+1])
            # 如果 attn_mask_down 全为 True, 则不 mask 掉, 确保一个 query 至少能查到一个点
            attn_mask_down[
                torch.where(attn_mask_down.sum(-1) == attn_mask_down.shape[-1])
            ] = False
        # 4. update query, concat + MLP
        query_feat = torch.cat([query_feat, cls_query, mask_query], dim=-1)
        query_feat = self.query_proj(query_feat)
        return masks_heatmap, cls_logits, attn_mask_down, query_feat

    def mask_head(self, scene_feat, query_feat, scene):
        """mask head
        - scene_feat: N_point x embedding_dim
        - query_feat: N_query x embedding_dim, after norm_before_head"""
        mask_feat = self.mask_head_mlp(query_feat)  # N_query x embedding_dim
        masks_logits = (
            scene_feat @ mask_feat.T
        )  # embedding 相乘得到 N_query 个 mask, N_point x N_query
        masks_heatmap = masks_logits.sigmoid()  # element_wise 归一化
        # sdf refinement
        point_index, query_index = (
            (masks_heatmap > 0.5).nonzero().T
        )  # all instance point num N_all
        # ins_point = scene.points[-1][point_index]    # crop instance points, N_all x 3 ->? N_all x embedding_dim -> (N_ins, embedding_dim)
        # query_feat_select = query_feat[query_index]  # N_all x embedding_dim
        # # point feature mean (N_ins, embedding_dim)
        # sdf_feat = torch.cat([query_feat_select, ins_point], dim=-1)    # N_all x (embedding_dim+3)
        # sdf_logits = self.sdf_head(sdf_feat)    # N_all x 1
        # masks_heatmap[point_index, query_index] = torch.mul(masks_heatmap[point_index, query_index], sdf_logits.sigmoid())
        # 学的时候不是学完就固定，需要有 scene aware 的去约束，把它弄成一个 test data 相关的
        mask_query = masks_heatmap @ scene_feat.T  # N_query x embedding_dim
        mask_query = self.norm_before_head(mask_query)
        self.align_proto[query_index] = self.proto_head(
            torch.cat([self.align_proto[query_index], mask_query], dim=-1)
        )
        self.align_proto = self.norm_before_head(self.align_proto)
        return masks_heatmap, masks_heatmap

    def cls_head(self, query_feat):
        """cls head
        - query_feat: N_query x embedding_dim, after norm_before_head"""
        cls_feat = self.cls_head_mlp(query_feat)  # N_query x embedding_dim
        cls_logits = (
            cls_feat @ self.sem_proto.weight.T
        )  # N_query x num_classes, 1x1 conv
        return cls_logits

    def transformer_fwd(self, query: Query, scene: Scene, last_masks_heatmap=None):
        """
        transformer forward
        - query: query data dict
        - scene: N_point data dict
            - features: list, features[j] == (N_point[j], embedding_dim)
        - last_masks_heatmap: N_point[-1] x N_query, corresponding to query.feat and query.pe
            TODO mask_heatmap 在新增 query 时的 1 padding 也放到外面顶层 model
        """
        masks_heatmap = (
            last_masks_heatmap.clone().detach()
            if last_masks_heatmap is not None
            else None
        )
        attn_mask = masks_heatmap
        for j in range(self.features_num - self.enable_final_block):
            # (features_num - self.enable_final_block) 个 block
            scene_feat = scene.features[j]  # (N_point[j], embedding_dim)
            # TODO 需要对 feat 进行排序吗? feat and scene_pe must be corresponding in point-wise
            # scene_feat = self.in_proj_layers[j](scene_feat.float())   # (N_point[j], embedding_dim)
            scene_feat, query.feat = self.blocks[j](
                source=scene_feat,
                query=query.feat,
                source_pe=scene.pe_list[j],
                query_pe=query.pe,
                source_batch=scene.batch_list[j],
                query_batch=query.batch,
                attn_mask=attn_mask if self.with_attn_mask[j] else None,
            )
            masks_heatmap, cls_logits, attn_mask = self.mask_module(
                scene.features[-1], query.feat
            )

        return masks_heatmap, cls_logits

    def forward(self, query: Query, scene: Scene, last_masks_heatmap=None):
        """
        - query: query data dict
            - feat: N_query x embedding_dim
            - pe: positional encoding, N_query x embedding_dim
            - batch: (N_query,)
        - scene: N_point data dict
            - features: list, features[j] == (N_point[j], features_dims[j])
            - feature_pyramid: features collection layer
            - pe_list: list, pe_list[j] == (N_point[j], embedding_dim)
            - batch_list: list, batch_list[j] == (N_point[j],), 一般可以直接从 scene.feature_pyramid.features[j] 中获取
        - last_masks_heatmap: N_point[-1] x N_query, corresponding to query.feat and query.pe
        """
        scene.features = self.in_projection(scene.features)
        query.feat = self.in_proj_query(query.feat)
        self.align_proto = self.sem_proto.weight

        # 首次, 利用click embedding生成pred_logits
        if last_masks_heatmap == None:
            last_masks_heatmap, _ = self.head(scene.features[-1], query.feat)
        # with torch.cuda.amp.autocast(enabled=True):          # 用 flash attention 时, 必须auto cast 到低精度
        masks_heatmap, cls_logits = self.transformer_fwd(
            query, scene, last_masks_heatmap
        )
        return masks_heatmap, cls_logits
