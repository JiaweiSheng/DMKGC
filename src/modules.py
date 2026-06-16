"""Cross-graph attention fusion for DMKGC."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion_sum1(nn.Module):
    """Cross-KG multi-head attention fusion producing diffusion conditions."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dropout = nn.Dropout(args.dropout)
        self.num_kgs = args.num_kgs
        self.d_input = args.entity_dim

        assert self.d_input % self.n_heads == 0, "d_input must be divisible by n_heads"
        self.d_head = self.d_input // self.n_heads

        self.w_q = nn.Linear(self.d_input, self.d_input)
        self.w_k = nn.Linear(self.d_input, self.d_input)

        self.beta_prior = nn.Embedding(self.num_kgs, self.num_kgs)
        nn.init.ones_(self.beta_prior.weight)

    def forward(self, x_gnn_output_all, kg_index):
        """x_gnn_output_all: [num_kgs, batch_size, out_dim]"""
        batch_size = x_gnn_output_all.size(1)
        num_kgs = x_gnn_output_all.size(0)

        x_gnn_output_all = x_gnn_output_all.permute(1, 0, 2)

        q = self.w_q(x_gnn_output_all[:, kg_index:kg_index + 1, :])
        k = self.w_k(x_gnn_output_all)
        v = x_gnn_output_all

        q = q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, num_kgs, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, num_kgs, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        kg_index_tensor = torch.tensor([kg_index], device=x_gnn_output_all.device).long()
        beta_prior_val = self.beta_prior(kg_index_tensor).view(1, 1, 1, num_kgs)
        attn_scores = attn_scores * beta_prior_val

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.d_input).squeeze(1)

        if self.args.v_att == 'cxt':
            return context
        if self.args.v_att == 'cxt+cur':
            return context + x_gnn_output_all[:, kg_index, :]
        if self.args.v_att == 'cxt+mean':
            return context + x_gnn_output_all.mean(1)
        return context
