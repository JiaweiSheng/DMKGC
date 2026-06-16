"""DMKGC main model: GNN encoding, diffusion denoising, and TransE link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn import GNN
from src.utils import get_ent_id
from src.modules import AttentionFusion_sum1
from src.dm import diffusion, Tenc


class DMKGC(nn.Module):
    """Diffusion-enhanced multi-domain knowledge graph completion model."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_kgs = args.num_kgs
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = args.device
        self.criterion = nn.MarginRankingLoss(margin=args.margin, reduction='mean')

        self.entity_embedding_layer = nn.Embedding(self.num_entities, self.entity_dim)
        nn.init.xavier_uniform_(self.entity_embedding_layer.weight)

        self.rel_embedding_layer = nn.Embedding(self.num_relations, self.relation_dim)
        nn.init.xavier_uniform_(self.rel_embedding_layer.weight)

        # Relation prior used as GNN edge weights
        self.relation_prior = nn.Embedding(self.num_relations, 1)
        nn.init.xavier_uniform_(self.relation_prior.weight)

        self.encoder_KG = GNN(
            num_kgs=args.num_kgs,
            in_dim=args.entity_dim,
            in_edge_dim=args.relation_dim,
            n_hid=args.encoder_hdim_gnn,
            out_dim=args.entity_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers_gnn,
            dropout=args.dropout,
        )

        self.dm = diffusion(
            timesteps=args.n_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_sche=args.beta_sche,
            args=args,
        )
        self.denoise_model = Tenc(
            hidden_size=args.entity_dim,
            dropout=args.dropout,
            diffuser_type=args.diffuser_type,
            device=self.device,
        )

        self.AttentionFusion_sum = AttentionFusion_sum1(args)
        self.decoder = TransE(device=self.device)

    def forward_GNN_embedding(self, graph_input_list, kg_index, info_dict=None):
        """Encode multi-KG subgraphs and fuse cross-domain context via diffusion."""
        loss_recon_all = torch.tensor([0.]).cuda()
        loss_reg_all = torch.tensor([0.]).cuda()
        loss_dn_all = torch.tensor([0.]).cuda()
        x_gnn_output_all = []
        x_gnn_output = None

        for graph_input in graph_input_list:
            x_features = self.entity_embedding_layer(graph_input.x)
            edge_beta_r = self.relation_prior(graph_input.edge_attr)
            edge_relation_embedding = self.rel_embedding_layer(graph_input.edge_attr)

            x_gnn_output = self.encoder_KG(
                x_features,
                graph_input.edge_index,
                graph_input.edge_kg_index,
                edge_beta_r,
                edge_relation_embedding,
                graph_input.y,
                graph_input.num_size,
            )
            x_gnn_output_all.append(x_gnn_output)

        x_gnn_output_all = torch.stack(x_gnn_output_all)
        predicted_x_reg = torch.zeros_like(x_gnn_output_all.mean(0)).to(self.device)

        ent_ids = get_ent_id(graph_input_list[0])
        x_origin = self.entity_embedding_layer(ent_ids)
        bs = x_gnn_output.size(0)
        x = x_origin
        cond = self.AttentionFusion_sum(x_gnn_output_all, kg_index)

        if self.training:
            t = torch.randint(0, self.args.n_steps, (bs,), device=self.device).long()
            eps = torch.randn_like(x).to(self.device)
            h = self.denoise_model.cacu_h(cond, self.args.p_uncond)
            x_noisy = self.dm.q_sample(x_start=x, t=t, noise=eps)
            predicted_x = self.denoise_model(x_noisy, h, t)
            loss_recon_all += F.mse_loss(x.detach(), predicted_x)

            # Random KG condition for cross-domain regularization
            random_kg_indices = torch.randint(low=0, high=x_gnn_output_all.size(0), size=(bs,))
            cond_rd = x_gnn_output_all[random_kg_indices, torch.arange(bs), :]
            h_rd = self.denoise_model.cacu_h(cond_rd, self.args.p_uncond)
            predicted_x_rd = self.denoise_model(x_noisy, h_rd, t)
            loss_reg_all += F.mse_loss(predicted_x_rd, x.detach())
        else:
            eps = torch.randn_like(x).to(self.device)
            T = torch.full((bs,), self.args.n_sampling_step - 1, dtype=torch.long, device=self.device)
            x_T = self.dm.q_sample(x, T, noise=eps)
            predicted_x = self.denoise_model.predict(
                x_T, cond, diff=self.dm, n_sampling_step=self.args.n_sampling_step
            )

        x_out = x_gnn_output_all[kg_index] + predicted_x + x_origin
        return x_out, {
            'loss_recon': loss_recon_all,
            'loss_dn': loss_dn_all,
            'loss_reg': loss_reg_all,
            'predicted_x_reg': predicted_x_reg,
            'x_gnn_output_all': x_gnn_output_all,
        }

    def forward_kg(self, h_graph, sample, t_graph, t_neg_graph, kg_index, t_neg_id=None):
        """Compute link prediction loss and diffusion auxiliary losses for one batch."""
        h, h_experts = self.forward_GNN_embedding(h_graph, kg_index)
        r = self.rel_embedding_layer(sample[:, 1])
        t, t_experts = self.forward_GNN_embedding(t_graph, kg_index)
        t_neg, t_neg_experts = self.forward_GNN_embedding(t_neg_graph, kg_index)

        r = r.unsqueeze(1)
        h = h.unsqueeze(1)
        t = t.unsqueeze(1)
        t_neg = t_neg.unsqueeze(1)

        bs = h.size(0)
        target = torch.tensor([-1], dtype=torch.long, device=self.device)

        pos_loss, neg_loss = self.decoder(h, r, t, t_neg)
        kg_loss = self.criterion(
            pos_loss.expand(bs, bs).reshape(-1),
            neg_loss.transpose(-1, -2).expand(bs, bs).reshape(-1),
            target,
        )

        loss_recon = h_experts['loss_recon'] + t_experts['loss_recon'] + t_neg_experts['loss_recon']
        loss_dn = h_experts['loss_dn'] + t_experts['loss_dn'] + t_neg_experts['loss_dn']
        loss_reg = h_experts['loss_reg'] + t_experts['loss_reg'] + t_neg_experts['loss_reg']
        return {
            'kg_loss': kg_loss,
            'loss_recon': loss_recon,
            'loss_dn': loss_dn,
            'loss_reg': loss_reg,
        }

    def predict_r_embedding(self, r):
        return self.rel_embedding_layer(r)

    def predict(self, h_emb, r, z=None):
        return self.decoder.predict(h_emb, r, z)

    def predict_candidate(self, c, z=None):
        return self.decoder.predict_candidate(c, z)

    def predict_score_fuc(self):
        return self.decoder.define_score

    def predict_drz(self, r):
        return None

    def predict_dr(self, z):
        return None


class TransE(nn.Module):
    """TransE distance-based scoring decoder."""

    def __init__(self, device):
        super(TransE, self).__init__()
        self.device = device

    def project_t(self, hr):
        return hr[0] + hr[1]

    def define_score(self, t_true_pred, d_r=None):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        return torch.norm(t_true - t_pred + 1e-8, dim=2)

    def forward(self, h, r, t, t_neg):
        projected_t = self.project_t([h, r])
        pos_loss = self.define_score([t, projected_t])
        neg_loss = self.define_score([t_neg, projected_t])
        return pos_loss, neg_loss

    def predict(self, h_emb, r, z=None):
        h = h_emb.unsqueeze(1)
        r = r.unsqueeze(1)
        return self.project_t([h, r])

    def predict_candidate(self, c, z=None):
        return c.unsqueeze(0)
