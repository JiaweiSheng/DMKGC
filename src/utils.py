"""Graph construction, model persistence, and evaluation utilities."""

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


class MyData(Data):
    """PyG Data subclass: edge_kg_index is excluded from batch dimension inference."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_kg_index':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def save_model(model, output_dir, filename, args):
    """Save model weights and training arguments."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({'state_dict': model.state_dict(), 'args': args}, ckpt_path)


def get_negative_samples_graph(batch_size_each, num_entity):
    return torch.randint(high=num_entity, size=(batch_size_each,))


def ranking_all_batch(predicted_t, embedding_matrix, k=None, define_score=None, d_r=None):
    """Rank candidate entities in batch and return top-k indices and scores."""
    if define_score is not None:
        distance = define_score([predicted_t, embedding_matrix])
    else:
        distance = torch.norm(predicted_t - embedding_matrix, dim=2)

    if k is None:
        k = embedding_matrix.size(1)

    top_k_scores, top_k_indexes = torch.topk(-distance, k=k)
    return top_k_indexes, top_k_scores


def get_language_list(entity_dir):
    entity_files = sorted(os.listdir(entity_dir))
    return [e[:2] for e in entity_files]


def get_kg_edges_for_each(kg_dir, language):
    """Load train/val edges for one KG and build an undirected edge_index."""
    train_df = pd.read_csv(
        os.path.join(kg_dir, language + '-train.tsv'),
        sep='\t', header=None, names=['head', 'relation', 'tail'],
    )
    val_df = pd.read_csv(
        os.path.join(kg_dir, language + '-val.tsv'),
        sep='\t', header=None, names=['head', 'relation', 'tail'],
    )

    sender_node_list = train_df['head'].values.astype(int).tolist()
    sender_node_list += train_df['tail'].values.astype(int).tolist()
    receiver_node_list = train_df['tail'].values.astype(int).tolist()
    receiver_node_list += train_df['head'].values.astype(int).tolist()
    edge_relation_list = train_df['relation'].values.astype(int).tolist()
    edge_relation_list += train_df['relation'].values.astype(int).tolist()

    val_sender = val_df['head'].values.astype(int).tolist() + val_df['tail'].values.astype(int).tolist()
    val_receiver = val_df['tail'].values.astype(int).tolist() + val_df['head'].values.astype(int).tolist()
    val_relation = val_df['relation'].values.astype(int).tolist() + val_df['relation'].values.astype(int).tolist()

    sender_node_list += val_sender
    receiver_node_list += val_receiver
    edge_relation_list += val_relation

    edge_index = torch.LongTensor(np.vstack((sender_node_list, receiver_node_list)))
    edge_relation = torch.LongTensor(np.asarray(edge_relation_list))
    return edge_index, edge_relation


def get_all_edges(kg_dir, kg_objects_dict, all_entity_global_index):
    """Merge edges from all domain KGs and map entity IDs to the global space."""
    edge_index_list = []
    edge_relation_list = []

    def get_global(x, language):
        return all_entity_global_index[language][x]

    for language in kg_objects_dict:
        edge_index, edge_relation = get_kg_edges_for_each(kg_dir, language)
        edge_index_list.append(edge_index.apply_(lambda x: get_global(x, language)))
        edge_relation_list.append(edge_relation)

    return torch.cat(edge_index_list, dim=1), torch.cat(edge_relation_list, dim=0)


def create_subgraph_list(edge_index, edge_type, total_num_nodes, num_hops, k):
    """Build a k-hop subgraph per global entity with at most k edges."""
    subgraph_list = []
    num_edges = []

    for i in tqdm(range(total_num_nodes)):
        subgraph_node_ids, edge_index_each, node_position, edge_masks = k_hop_subgraph(
            [i], num_hops, edge_index, num_nodes=total_num_nodes, relabel_nodes=True,
        )

        edge_index_each = edge_index_each[:, :k]
        edge_type_masked = (edge_type + 1) * edge_masks
        edge_attr = edge_type_masked[edge_type_masked.nonzero(as_tuple=True)] - 1
        edge_attr = edge_attr[:k]
        assert edge_attr.shape[0] == edge_index_each.shape[1]

        subgraph_each = Data(
            x=subgraph_node_ids,
            edge_index=edge_index_each,
            edge_attr=edge_attr,
            y=torch.LongTensor([node_position]),
            num_size=torch.LongTensor([len(subgraph_node_ids)]),
        )
        subgraph_list.append(subgraph_each)
        num_edges.append(edge_index_each.shape[1])

    print('Average subgraph edges %.2f' % np.mean(num_edges))
    return subgraph_list


def get_k_subgraph_list(subgraph_list, node_index, kg_index, num_kgs, entity2kgidx, total_num_nodes, data_dir):
    """Build num_kgs domain-view subgraphs per entity and cache them to disk."""
    k_subgraph_list_path = os.path.join(data_dir, f'kg{kg_index}_k_subgraph_list.graph')
    if os.path.exists(k_subgraph_list_path):
        return torch.load(k_subgraph_list_path)

    def get_kg_index(x, subset, node2kgidx):
        return node2kgidx[subset[x].item()]

    k_subgraph_list = []
    print('get_k_subgraph_list: kg_index:', kg_index)

    for i in tqdm(node_index):
        graphs = []
        subgraph = subgraph_list[i]
        nodes = subgraph.x
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr

        for j in range(num_kgs):
            subset_j = [i]
            tmp_node2kgidx = {i: kg_index}
            edge_mask = edge_index.new_empty((2, edge_index.shape[1]), dtype=torch.bool)
            edge_mask.fill_(False)

            for r in range(2):
                for c in range(edge_index.shape[1]):
                    node_rc = nodes[edge_index[r][c]]
                    if j in entity2kgidx[node_rc.item()] or node_rc == i:
                        subset_j.append(node_rc.item())
                        edge_mask[r][c] = True
                        if node_rc != i:
                            tmp_node2kgidx[node_rc.item()] = j

            subset_j, inv = torch.tensor(subset_j).unique(return_inverse=True)
            inv = inv[:1]
            edge_mask = edge_mask[0] & edge_mask[1]
            edge_index_j = nodes[edge_index[:, edge_mask]]
            tmp_index = edge_index_j.new_full((total_num_nodes,), -1)
            tmp_index[subset_j] = torch.arange(subset_j.size(0))
            edge_index_j = tmp_index[edge_index_j]
            edge_attr_j = edge_attr[edge_mask]
            num_size_j = torch.LongTensor([len(subset_j)])

            edge_kg_index = copy.deepcopy(edge_index_j)
            edge_kg_index.apply_(lambda x: get_kg_index(x, subset_j, tmp_node2kgidx))
            graphs.append(MyData(
                x=subset_j, edge_index=edge_index_j, edge_kg_index=edge_kg_index,
                edge_attr=edge_attr_j, y=inv, num_size=num_size_j,
            ))

        k_subgraph_list.append(graphs)

    torch.save(k_subgraph_list, k_subgraph_list_path)
    return k_subgraph_list


def nodes_to_k_graph(k_subgraph_list, node_index, device, shuffle=False):
    """Pack per-KG subgraphs for entities in a batch into PyG batches."""
    batch_size = node_index.shape[0]
    graph_batches = []

    for i in range(len(k_subgraph_list[0])):
        graphs = [k_subgraph_list[j.item()][i] for j in node_index]
        graph_loader = GraphLoader(graphs, batch_size=batch_size, shuffle=shuffle)
        for batch in graph_loader:
            graph_batches.append(batch.to(device))

    return graph_batches


def get_ent_id(graph_input):
    """Extract global IDs of center entities from a batched subgraph."""
    y = graph_input.y
    s = graph_input.num_size
    x = graph_input.x
    ent_ids = []
    node_base = 0

    for i in range(y.shape[0]):
        ent_ids.append(x[node_base + y[i]].view(-1, 1))
        node_base += s[i]

    return torch.cat(ent_ids).to(y.device).view(-1)
