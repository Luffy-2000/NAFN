"""
Transformer-based prototype aggregation (TraNFS-style).
作为降噪方法：用 attention 对 support 做加权聚合，输出 prototypes 和 sample_weights.
"""
import torch
import torch.nn as nn


class TransformerProto(nn.Module):
    """
    TraNFS 风格：cls tokens + support embeddings，Transformer 聚合。
    输入: s_embeddings (n_support, dim), support_labels, way, shot
    输出: prototypes (way, dim), sample_weights (n_support,)
    """

    def __init__(self, d_model=320, nhead=8, num_layers=2, max_way=20, max_shot=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_way = max_way
        self.max_shot = max_shot

        self.cls_embeddings = nn.Embedding(max_way, d_model)
        self.pos_embeddings = nn.Embedding(max_way, d_model)  # 按 class id，way 个位置

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, s_embeddings, support_labels, way, shot):
        """
        Args:
            s_embeddings: (n_support, d_model), n_support = way * shot
            support_labels: (n_support,)
            way, shot: 当前 episode 的 way 和 shot
        Returns:
            prototypes: (way, d_model)
            sample_weights: (n_support,) - 用于 LR 的样本权重
        """
        device = s_embeddings.device
        n_support = s_embeddings.size(0)
        assert n_support == way * shot

        n_arng = torch.arange(way, device=device)
        cls_tokens = self.cls_embeddings(n_arng)
        seq = torch.cat([cls_tokens, s_embeddings], dim=0)

        pos_idx = torch.cat([
            n_arng,
            torch.repeat_interleave(n_arng, shot, output_size=n_support)
        ])
        pos_tokens = self.pos_embeddings(pos_idx)
        seq = seq + pos_tokens

        seq = seq.unsqueeze(0)
        output, attn_weights = self._forward_with_attn(seq)

        output = output.squeeze(0)
        prototypes = output[:way]

        attn = attn_weights[-1]
        if attn.dim() == 4:
            attn = attn.mean(dim=1).squeeze(0)
        elif attn.dim() == 3:
            attn = attn.squeeze(0)
        sample_weights = torch.zeros(n_support, device=device, dtype=attn.dtype)
        for i in range(n_support):
            c = support_labels[i].item()
            sample_weights[i] = float(attn[c, way + i].mean())
        sample_weights = sample_weights + 1e-8
        sample_weights = sample_weights / sample_weights.sum() * n_support

        return prototypes, sample_weights

    def _forward_with_attn(self, src):
        output = src
        attn_list = []
        for layer in self.encoder.layers:
            src2, attn = layer.self_attn(
                output, output, output,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=True,
            )
            output = output + layer.dropout1(src2)
            output = layer.norm1(output)
            output = output + layer._ff_block(output)
            output = layer.norm2(output)
            attn_list.append(attn)
        return output, attn_list
