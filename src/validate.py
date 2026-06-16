"""Link prediction evaluation: Hits@K, MRR, and filtered setting."""

import logging
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import nodes_to_k_graph, ranking_all_batch


def hr2t_from_train_set(data_dir, target_lang):
    """Build (head, relation) -> tails mapping for filtered evaluation."""
    hr2t = {}

    for split in ('train', 'val'):
        df = pd.read_csv(
            os.path.join(data_dir, f'{target_lang}-{split}.tsv'),
            sep='\t', header=None,
        )
        for h, r, t in df.values.astype(int):
            key = (int(h), int(r))
            hr2t.setdefault(key, set()).add(int(t))

    return hr2t


class Tester:
    """Evaluate the model on validation or test splits."""

    def __init__(self, args, kg_objects_dict, model, device, data_dir):
        self.args = args
        self.kg_objects_dict = kg_objects_dict
        self.model = model
        self.device = device
        self.data_dir = data_dir

    def test(self, is_val=True, is_filtered=False):
        results = {}
        for kg_name in self.kg_objects_dict:
            results[kg_name] = self.test_kg(kg_name, is_val, is_filtered)
        return results

    def test_kg(self, kg_name, is_val=True, is_filtered=False):
        time0 = time.time()
        kg = self.kg_objects_dict[kg_name]

        if is_val:
            h, r, t = kg.h_val, kg.r_val, kg.t_val
            output_text = f'[{kg_name}] Val:'
        else:
            h, r, t = kg.h_test, kg.r_test, kg.t_test
            output_text = f'[{kg_name}] Test:'

        num_samples = h.shape[0]
        ground_truth = t.view(-1, 1).to(self.device)
        kg_batch_generator = kg.generate_batch_data(
            h, r, t, batch_size=self.args.test_batch_size, shuffle=False,
        )
        ground_truth_generator = DataLoader(
            ground_truth, batch_size=self.args.test_batch_size, shuffle=False,
        )

        self.pre_compute_all_embeddings(kg_name)

        if is_filtered and not is_val:
            hr2t_train = hr2t_from_train_set(self.data_dir + '/kg', kg_name)

            def hr2t_filter(e, head, rel):
                return -1 if e in hr2t_train[(head, rel)] else e

        hits_1_compute, hits_10_compute, rranks_sum = 0, 0, 0
        for kg_batch, ground_truth_batch in zip(kg_batch_generator, ground_truth_generator):
            h_batch = kg_batch[:, 0]
            r_batch = kg_batch[:, 1].to(self.device)
            h_embedding = kg.computed_entity_embedding_kg[h_batch, :]
            r_embedding = self.model.predict_r_embedding(r_batch)
            drz = self.model.predict_drz(r_batch)

            model_predictions = self.model.predict(h_embedding, r_embedding, z=drz)
            candidate_emb = self.model.predict_candidate(kg.computed_entity_embedding_kg, z=drz)
            dr = self.model.predict_dr(z=drz)

            ranking_indices, ranking_scores = ranking_all_batch(
                model_predictions, candidate_emb,
                define_score=self.model.predict_score_fuc(), d_r=dr,
            )

            if is_filtered and not is_val:
                for i in range(h_batch.shape[0]):
                    head = h_batch[i].item()
                    rel = r_batch[i].item()
                    if (head, rel) in hr2t_train:
                        ranking_indices[i, :] = ranking_indices[i, :].cpu().apply_(
                            lambda e: hr2t_filter(e, head, rel),
                        ).to(self.device)
                        p_1 = (ranking_indices[i, :] == -1).to(torch.long)
                        _, idx = p_1.sort()
                        ranking_indices[i, :] = ranking_indices[i, :].gather(0, idx)

            batch_hits_1, batch_hits_10, batch_rranks_sum = self.get_hit_mrr(
                ranking_indices, ground_truth_batch,
            )
            hits_1_compute += batch_hits_1
            hits_10_compute += batch_hits_10
            rranks_sum += batch_rranks_sum

        kg.computed_entity_embedding_kg = None

        hits_1_ratio = hits_1_compute / num_samples
        hits_10_ratio = hits_10_compute / num_samples
        mrr = rranks_sum / num_samples

        if is_filtered and not is_val:
            logging.info('{} filterd: {:.4f}, {:.4f}, {:.4f}'.format(
                output_text, hits_1_ratio, hits_10_ratio, mrr,
            ))
        else:
            logging.info('{} {:.4f}, {:.4f}, {:.4f}'.format(
                output_text, hits_1_ratio, hits_10_ratio, mrr,
            ))

        logging.info('time:{},{},{},{}'.format(
            kg_name, time.time() - time0, num_samples, (time.time() - time0) / num_samples,
        ))
        return [hits_1_ratio, hits_10_ratio, mrr]

    def pre_compute_all_embeddings(self, kg_name):
        """Pre-compute GNN embeddings for all entities in the current KG."""
        with torch.no_grad():
            kg = self.kg_objects_dict[kg_name]
            kg_index = self.args.kgname2idx[kg_name]
            node_index_tensor = torch.arange(kg.num_entities)
            dataloader = DataLoader(node_index_tensor, batch_size=self.args.test_batch_size, shuffle=False)
            embedding_list = []

            for data in dataloader:
                graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, data, self.args.device)
                node_embeddings, _ = self.model.forward_GNN_embedding(
                    graph_batch_list, kg_index, data.to(self.device),
                )
                embedding_list.append(node_embeddings)

            self.kg_objects_dict[kg_name].computed_entity_embedding_kg = torch.cat(embedding_list, dim=0)

    def get_hit_mrr(self, topk_indices_all, ground_truth):
        zero_tensor = torch.tensor([0]).to(ground_truth.device)
        one_tensor = torch.tensor([1]).to(ground_truth.device)

        hits_1 = torch.where(ground_truth == topk_indices_all[:, :1], one_tensor, zero_tensor).sum().item()
        hits_10 = torch.where(ground_truth == topk_indices_all[:, :10], one_tensor, zero_tensor).sum().item()

        gt_expanded = ground_truth.expand_as(topk_indices_all)
        hits = (gt_expanded == topk_indices_all).nonzero()
        ranks = hits[:, -1] + 1
        rranks_sum = torch.sum(torch.reciprocal(ranks.float()))
        return hits_1, hits_10, rranks_sum
