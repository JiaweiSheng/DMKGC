"""Multi-domain graph-attention GNN encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add, scatter_max


def softmax(src, index, num_nodes=None):
    """Softmax-normalize multi-head attention weights per destination node."""
    out = []
    for s in src:
        out_s = s - scatter_max(s, index, dim=0, dim_size=num_nodes)[0][index]
        out_s = out_s.exp()
        out_s = out_s / (scatter_add(out_s, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
        out.append(out_s)
    return torch.stack(out)


class MGA(MessagePassing):
    """Multi-graph Attention layer with KG-specific linear transforms."""

    def __init__(self, num_kgs=1, n_heads=2, d_input=32, d_input_edge=32, d_out=32, dropout=0.1):
        super().__init__(aggr='add')
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.d_input = d_input
        self.d_q = d_out // n_heads
        self.d_k = d_out // n_heads
        self.d_v = d_out // n_heads
        self.d_sqrt = math.sqrt(d_out // n_heads)

        self.w_q_list = nn.ModuleList()
        self.w_k_list = nn.ModuleList()
        self.w_v_list = nn.ModuleList()
        self.w_transfer_list = nn.ModuleList()
        for _ in range(num_kgs):
            self.w_q_list.append(nn.Linear(self.d_input, self.d_q * n_heads, bias=True))
            self.w_k_list.append(nn.Linear(self.d_input, self.d_k * n_heads, bias=True))
            self.w_v_list.append(nn.Linear(self.d_input, self.d_v * n_heads, bias=True))
            self.w_transfer_list.append(nn.Linear(self.d_input + d_input_edge, self.d_input, bias=True))

        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding):
        num_nodes = x.shape[0]
        residual = x
        x = self.layer_norm(x)
        return self.propagate(
            edge_index, x=x, edge_kg_index=edge_kg_index, edge_beta_r=edge_beta_r,
            edge_relation_embedding=edge_relation_embedding, residual=residual, num_nodes=num_nodes,
        )

    def message(self, x_j, x_i, edge_index_i, edge_kg_index, edge_beta_r, edge_relation_embedding, num_nodes):
        edge_kg_index_j = edge_kg_index[0]
        edge_kg_index_i = edge_kg_index[1]
        edge_value = edge_beta_r.view(-1, 1)

        x_j_transfer = F.gelu(self.compute_transfer(
            torch.cat([x_j, edge_relation_embedding], dim=1),
            self.w_transfer_list, self.d_input, edge_kg_index_j,
        ))

        attention = self.multi_head_cross_attention(
            x_i, x_j_transfer, edge_value, edge_kg_index_i, edge_kg_index_j,
        )
        attention = torch.div(attention, self.d_sqrt)
        attention_norm = softmax(attention, edge_index_i, num_nodes)

        sender = x_j_transfer.view(-1, self.n_heads, self.d_v).transpose(0, 1)
        message = attention_norm * sender
        return message.transpose(0, 1).reshape(-1, self.d_v * self.n_heads)

    def compute_transfer(self, input_x, linear_transfer, output_dim, input_edge_kg_index):
        """Apply KG-specific linear transforms to edge features."""
        out_x = input_x.new_empty((input_x.shape[0], output_dim))
        for i in range(len(linear_transfer)):
            idx_i = torch.nonzero(input_edge_kg_index == i).view(-1)
            out_x[idx_i] = linear_transfer[i](input_x[idx_i])
        return out_x

    def multi_head_cross_attention(self, x_i, x_j_transfer, edge_value, edge_kg_index_i, edge_kg_index_j):
        x_i = self.compute_transfer(x_i, self.w_q_list, self.d_q * self.n_heads, edge_kg_index_i)
        x_i = x_i.view(-1, self.n_heads, self.d_q).transpose(0, 1)
        x_j = self.compute_transfer(x_j_transfer, self.w_k_list, self.d_k * self.n_heads, edge_kg_index_j)
        x_j = x_j.view(-1, self.n_heads, self.d_k).transpose(0, 1)

        attention = torch.matmul(torch.unsqueeze(x_j, dim=2), torch.unsqueeze(x_i, dim=3))
        edge_value = torch.unsqueeze(edge_value, dim=2)
        attention = attention * edge_value
        return torch.squeeze(attention, dim=2)

    def update(self, aggr_out, residual):
        return self.dropout(residual + F.gelu(aggr_out))


class GNN(nn.Module):
    """Stack MGA layers and extract target node representations from a batch."""

    def __init__(self, num_kgs, in_dim, in_edge_dim, n_hid, out_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            MGA(
                num_kgs=num_kgs, n_heads=n_heads, d_input=in_dim,
                d_input_edge=in_edge_dim, d_out=out_dim, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding, y=None, s=None):
        h_t = x
        for layer in self.gnn_layers:
            h_t = layer(h_t, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding)

        if y is not None:
            true_indexes = self.get_real_index(y, s)
            h_t = torch.index_select(h_t, 0, true_indexes)
        return h_t

    def get_real_index(self, y, s):
        """Map local center-node indices in each subgraph to batched global indices."""
        num_graphs = y.shape[0]
        node_base = torch.LongTensor([0]).to(y.device)
        true_index = []

        for i in range(num_graphs):
            true_index.append((node_base + y[i]).view(-1, 1))
            node_base += s[i]

        return torch.cat(true_index).to(y.device).view(-1)
