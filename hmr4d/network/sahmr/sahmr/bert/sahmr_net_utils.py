import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertLayer, BertPooler
from .modeling_bert import BertLayerNorm as LayerNormClass

from hmr4d.network.base_arch.attention.linear_attention import FullAttention, LinearAttention


class LoFTREncoderLayer(nn.Module):
    """https://github.com/zju3dv/LoFTR"""

    def __init__(self, d_model, nhead, attn_type, dropout=0.0):
        super().__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attn_type == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        Returns:
            message: [N, L, C]
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        if self.use_dropout:
            message = self.dropout1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        if self.use_dropout:
            message = self.dropout2(message)

        return message


class BertEncoder_CP(nn.Module):
    def __init__(self, config, cpose_cfg):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.cp_mimic_bert = cpose_cfg.get("cp_mimic_bert", "self_attn_linearx2")
        assert self.cp_mimic_bert == "self_attn_linearx2"
        self.linear_cp = nn.ModuleList(
            [
                LoFTREncoderLayer(config.hidden_size, config.num_attention_heads, "linear"),
                LoFTREncoderLayer(config.hidden_size, config.num_attention_heads, "linear"),
            ]
        )

        attn_type = cpose_cfg.attn_type  # cross-attention
        dropout = cpose_cfg.dropout
        self.loftr_layer = LoFTREncoderLayer(config.hidden_size, config.num_attention_heads, attn_type, dropout)

    def forward(self, hidden_states, feat_cp, feat_cp_mask, attention_mask, head_mask=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, head_mask[i])[0]

        # mimic bert network
        assert self.cp_mimic_bert == "self_attn_linearx2"
        message = self.linear_cp[0](feat_cp, feat_cp, source_mask=feat_cp_mask)
        feat_cp = feat_cp + message
        message = self.linear_cp[1](feat_cp, feat_cp, source_mask=feat_cp_mask)
        feat_cp = feat_cp + message

        # pass message to hidden_states
        message = self.loftr_layer(hidden_states, feat_cp, source_mask=feat_cp_mask)
        hidden_states = hidden_states + message

        return hidden_states, feat_cp


class CPOSE_Encoder(BertPreTrainedModel):
    def __init__(self, config, cpose_cfg):
        super(CPOSE_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder_CP(config, cpose_cfg)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.img_dim = config.img_feature_dim

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        img_feats,
        cp_feats,
        cp_mask,
        corr_pids,
        pid_to_verts431,
        cp_xyz,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
    ):
        B, V = img_feats.shape[:2]
        input_ids = torch.zeros([B, V], dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(V, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            # if head_mask.dim() == 1:
            #     head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            #     head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            # elif head_mask.dim() == 2:
            #     # We can specify head_mask for each layer
            #     head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)
        cp_embedding_output = self.img_embedding(cp_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output
        if corr_pids == None:
            cp_embeddings = position_embeddings + cp_embedding_output
        else:
            # 以求平均的方式获得part的embedding token，然后加给对应的part
            part_token = torch.stack(
                [position_embeddings[:, pid_to_verts431[i]].mean(1) for i in range(1, 8)], dim=1
            )  # (B, 7, E)
            part_token = F.pad(part_token, (0, 0, 1, 0), value=0.0)  # (B, 1+7, E)
            expand_part_token = torch.stack([part_token[b, corr_pids[b]] for b in range(B)])  # (B, 500, E)
            cp_embeddings = expand_part_token + cp_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
            cp_embeddings = self.LayerNorm(cp_embeddings)
        embeddings = self.dropout(embeddings)
        cp_embeddings = self.dropout(cp_embeddings)

        img_feats_out, cp_feats_out = self.encoder(
            embeddings, cp_embeddings, cp_mask, extended_attention_mask, head_mask=head_mask
        )

        return img_feats_out, cp_feats_out


class CPOSE(BertPreTrainedModel):
    """
    The archtecture of a transformer encoder block we used in METRO
    """

    def __init__(self, config, cpose_cfg):
        super(CPOSE, self).__init__(config)  # cpose_cfg is not part of BertPreTrainedModel
        self.config = config
        self.bert = CPOSE_Encoder(config, cpose_cfg)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)

        self.apply(self.init_weights)

    def forward(self, img_feats, cp_feats, cp_mask, corr_pids, pid_to_verts431, cp_xyz):
        """
        img_feats: (B, S, D)
        cp_feats: (B, T, D)
        """
        img_feats_out, cp_feats_out = self.bert(img_feats, cp_feats, cp_mask, corr_pids, pid_to_verts431, cp_xyz)

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        img_feats_final = self.cls_head(img_feats_out) + self.residual(img_feats)
        cp_feats_final = self.cls_head(cp_feats_out) + self.residual(cp_feats)

        return img_feats_final, cp_feats_final
