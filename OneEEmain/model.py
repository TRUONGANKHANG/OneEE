import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from transformers import AutoConfig, AutoModel, AutoTokenizer

import config


def load_pretrained_backbone(model_name, device, use_fast_tokenizer=False):
    """Load a pretrained backbone (AutoModel) and tokenizer. Returns (model, tokenizer)."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_name, config=cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    model.to(device)
    return model, tokenizer

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class Predictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, dropout=0.):
        super().__init__()
        self.mlp_sub = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp_obj = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        ent_sub = self.dropout(self.mlp_sub(x))
        ent_obj = self.dropout(self.mlp_obj(y))

        outputs = self.biaffine(ent_sub, ent_obj)

        return outputs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x



class FFNN(nn.Module):
    def __init__(self, input_dim, hid_dim, cls_num, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, hid_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.q_linear = nn.Linear(hid_size, hid_size)
        self.k_linear = nn.Linear(hid_size, hid_size * 2)

        self.factor = math.sqrt(hid_size)

        self.gate1 = Gate(hid_size, dropout=dropout)
        self.gate2 = Gate(hid_size, dropout=dropout)

    def forward(self, x, s, g):
        # x [B, L, H]
        # s [B, K, H]
        # g [B, N, H]
        # x = self.dropout(x)
        # s = self.dropout(s)
        q = self.q_linear(x)
        k_v = self.k_linear(g)
        k, v = torch.chunk(k_v, chunks=2, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.factor
        # scores = self.dropout(scores)
        scores = torch.softmax(scores, dim=-1)
        g = torch.bmm(scores, v)
        g = g.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        h = x.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        s = s.unsqueeze(1).expand(-1, x.size(1), -1, -1)

        h = self.gate1(h, g)
        h = self.gate2(h, s)
        return h


class Gate(nn.Module):
    def __init__(self, hid_size, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(hid_size, hid_size)

    def forward(self, x, y):
        '''
        :param x: B, L, K, H
        :param y: B, L, K, H
        :return:
        '''
        o = torch.cat([x, y], dim=-1)
        o = self.dropout(o)
        gate = self.linear(o)
        gate = torch.sigmoid(gate)
        o = gate * x + (1 - gate) * y
        # o = F.gelu(self.linear2(self.dropout(o)))
        return o


class Model(nn.Module):
    """
    OneEE-style span pointer model, backbone = PhoBERT (hoặc BERT bất kỳ).
    - tri_logits: [B, L, L, E]           (E = tri_label_num)
    - arg_logits: [B, L, L, E]
    - role_logits:[B, L, L, E, R]       (R = rol_label_num)
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.tri_label_num = config.tri_label_num   # E
        self.rol_label_num = config.rol_label_num  # R
        self.tri_hid_size = config.tri_hid_size
        self.arg_hid_size = config.arg_hid_size
        self.eve_hid_size = config.eve_hid_size

        # PhoBERT backbone (hoặc model nào em set trong config.bert_name)
        self.bert = AutoModel.from_pretrained(
            config.bert_name,
            cache_dir="./cache/"
        )

        self.dropout = nn.Dropout(config.dropout)

        # Mỗi head = 1 event-type (OneEE: multi-head pointer)
        # => output_dim = 2 * tri_hid_size * tri_label_num
        self.tri_linear = nn.Linear(
            config.bert_hid_size,
            self.tri_hid_size * 2 * self.tri_label_num
        )
        self.arg_linear = nn.Linear(
            config.bert_hid_size,
            self.arg_hid_size * 2 * self.tri_label_num
        )
        
        # --- role head nhẹ: không nhân theo event ---
        self.role_hid_size = config.eve_hid_size
        self.role_linear_q = nn.Linear(config.bert_hid_size, self.role_hid_size)
        self.role_linear_k = nn.Linear(config.bert_hid_size, self.role_hid_size)
        # biaffine cho R roles
        self.role_biaffine = Biaffine(n_in=self.role_hid_size, n_out=self.rol_label_num, bias_x=True, bias_y=True)

    def _sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        """
        y chang code cũ của em, chỉ dùng cho relative position trong pointer
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1).to(next(self.parameters()).device)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(position_ids.device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings

    def _pointer(self, qw, kw, word_mask2d):
        """
        qw, kw: [B, L, H, D] (H = num_heads, D = hid_size)
        word_mask2d: [B, L, L] (1 = hợp lệ)
        Trả logits: [B, H, L, L]
        """
        B, L, H, D = qw.size()
        pos_emb = self._sinusoidal_position_embedding(B, L, D)  # [B, L, D]
        pos_emb = pos_emb.to(qw.device)

        # cos_pos, sin_pos: [B, L, 1, D]
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)

        # Rotary-like rotation
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos

        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        # einsum: qw:[B,M,H,D], kw:[B,N,H,D] -> [B,H,M,N]
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        grid_mask2d = word_mask2d.unsqueeze(1).expand(B, H, L, L).float()
        logits = logits * grid_mask2d - (1 - grid_mask2d) * 1e12
        return logits  # [B,H,L,L]

    def forward(self,
            inputs,
            att_mask,
            word_mask1d,
            word_mask2d,
            triu_mask2d,
            tri_labels=None,
            arg_labels=None,
            role_labels=None,
            event_idx=None):
        """
        Train: trả về (tri_logits, arg_logits, role_logits)
        Eval : trả dict {"ti","tc","ai","ac"}
        """
        outputs = {}
        device = inputs.device

        L = word_mask1d.size(1)

        # ===== BERT / PhoBERT encoder =====
        bert_out = self.bert(input_ids=inputs, attention_mask=att_mask)
        if hasattr(bert_out, "last_hidden_state"):
            bert_embs = bert_out.last_hidden_state      # [B,seq,H]
        else:
            bert_embs = bert_out[0]

        # bỏ [CLS], cắt theo số token gốc
        bert_embs = bert_embs[:, 1:1 + L, :]           # [B,L,H]
        B, L, H = bert_embs.size()

        x = self.dropout(bert_embs)

        # ===== Trigger head =====
        tri_reps = self.tri_linear(x)                  # [B,L,2*H_tri*E]
        tri_reps = tri_reps.view(B, L, self.tri_label_num, self.tri_hid_size * 2)
        tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)          # [B,L,E,H_tri]

        tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d)    # [B,E,L,L]
        tri_logits = tri_logits.permute(0, 2, 3, 1)                # [B,L,L,E]

        # ===== Argument head =====
        arg_reps = self.arg_linear(x)                              # [B,L,2*H_arg*E]
        arg_reps = arg_reps.view(B, L, self.tri_label_num, self.arg_hid_size * 2)
        arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)          # [B,L,E,H_arg]

        arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d)    # [B,E,L,L]
        arg_logits = arg_logits.permute(0, 2, 3, 1)                # [B,L,L,E]

        # ===== Role head (nhẹ) =====
        # Lấy representation riêng cho trigger token và argument token
        role_q = self.role_linear_q(x)   # [B,L,H_r]
        role_k = self.role_linear_k(x)   # [B,L,H_r]

        # Biaffine: [B,R,L,L]
        role_logits = self.role_biaffine(role_q, role_k)  # B x R x L x L
        # role_logits = role_logits.permute(0, 2, 3, 1)      # [B,L,L,R]
        # ================= TRAIN =================
        if self.training:
            return tri_logits, arg_logits, role_logits

        # ================= EVAL ==================
        word_mask2d = word_mask2d.to(device)
        triu_mask2d = triu_mask2d.to(device)

        # prob + mask
        tri_prob = torch.sigmoid(tri_logits) * word_mask2d.unsqueeze(-1)   # [B,L,L,E]
        arg_prob = torch.sigmoid(arg_logits) * word_mask2d.unsqueeze(-1)   # [B,L,L,E]
        role_prob = torch.sigmoid(role_logits) * triu_mask2d.unsqueeze(-1)         # [B,L,L,R]

        # dynamic threshold theo độ “tự tin” của trigger
        max_p = tri_prob.max().item() if tri_prob.numel() > 0 else 0.0
        if max_p < 0.01:
            thr = 0.001
        elif max_p < 0.1:
            thr = 0.01
        else:
            thr = 0.05


        # lấy các span > threshold
        tri_b, tri_x, tri_y, tri_e = (tri_prob > thr).nonzero(as_tuple=True)
        arg_b, arg_x, arg_y, arg_e = (arg_prob > thr).nonzero(as_tuple=True)
        role_b, role_t, role_a, role_r = (role_prob > thr).nonzero(as_tuple=True)

        outputs["ti"] = torch.stack([tri_b, tri_x, tri_y], dim=1).cpu().numpy()
        outputs["tc"] = torch.stack([tri_b, tri_x, tri_y, tri_e], dim=1).cpu().numpy()
        outputs["ai"] = torch.stack([arg_b, arg_x, arg_y, arg_e], dim=1).cpu().numpy()
        outputs["ac"] = None   # rất quan trọng: decode() sẽ bỏ qua role
        # ===== Build AC: (b, arg_pos, event_type, role) =====
        # Map (b, arg_token) -> list of (x,y,e)
        # arg_dict = {}
        # for bb, xs, ys, ee in zip(arg_b.tolist(), arg_x.tolist(), arg_y.tolist(), arg_e.tolist()):
        #     for c in range(xs, ys + 1):
        #         arg_dict.setdefault((bb, c), []).append((xs, ys, ee))

        # ac_list = []
        # # duyệt qua tất cả role prediction (b, t, a, r)
        # for bb, tt, aa, rr in zip(role_b.tolist(), role_t.tolist(), role_a.tolist(), role_r.tolist()):
        #     key = (bb, aa)
        #     if key not in arg_dict:
        #         continue
        #     # với mỗi argument span (xs,ys,e) chứa token 'aa':
        #     for xs, ys, ee in arg_dict[key]:
        #         # chỉ nhận (e,r) hợp lệ theo tri_args
        #         if (ee, rr) in config.tri_args:
        #             ac_list.append([bb, aa, ee, rr])

        # if len(ac_list) > 0:
        #     outputs["ac"] = torch.tensor(ac_list, dtype=torch.long).cpu().numpy()
        # else:
        #     outputs["ac"] = np.empty((0, 4), dtype=np.int64)
        return outputs



class MyModel(nn.Module):
    def __init__(self, pretrained_name, device):
        super(MyModel, self).__init__()
        self.device = device
        self.backbone, self.tokenizer = load_pretrained_backbone(pretrained_name, device)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 3)  # ví dụ

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
