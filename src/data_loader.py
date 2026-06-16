"""Multi-domain knowledge graph dataset loading and preprocessing."""

import os

import numpy as np
import pandas as pd
import torch

from src.knowledgegraph import KnowledgeGraph
from src.utils import get_all_edges, create_subgraph_list


class MKGDataset:
    """Load multiple domain KGs, alignment links, and global subgraph structures."""

    def __init__(self, args, MAX_SAM=10000000000):
        self.data_dir = args.data_path + args.dataset
        self.entity_dir = self.data_dir + '/entity'
        self.kg_dir = self.data_dir + '/kg'
        self.align_dir = self.data_dir + '/seed_alignlinks'
        self.args = args
        self.MAX_SAM = MAX_SAM
        self.kg_names = args.langs
        self.num_kgs = len(self.kg_names)

    def load_data(self):
        return self.create_KG_objects_and_subgraph()

    def create_KG_objects_and_subgraph(self):
        kg_objects_dict = {}
        seeds = self.load_align_links()
        pre_langs = []
        all_entity2kgs = {}
        all_entity_global_index = {}

        for lang in self.kg_names:
            kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations = self.load_kg_data(lang)
            kg_object = KnowledgeGraph(
                lang, kg_train_data, kg_val_data, kg_test_data,
                num_entities, num_relations, self.args.device,
            )
            kg_object.get_global_h_t(seeds, pre_langs, all_entity2kgs, all_entity_global_index)
            kg_objects_dict[lang] = kg_object

        self.num_entities = len(all_entity2kgs)
        self.num_relations = num_relations

        edge_index, edge_type = get_all_edges(self.kg_dir, kg_objects_dict, all_entity_global_index)
        graph_dir = '_'.join(self.kg_names)
        subgraph_list_path = os.path.join(self.data_dir, graph_dir, 'subgraph_list.graph')

        if not os.path.exists(subgraph_list_path):
            os.makedirs(os.path.join(self.data_dir, graph_dir), exist_ok=True)
            subgraph_list = create_subgraph_list(
                edge_index, edge_type, self.num_entities, self.args.num_hops, self.args.k,
            )
            torch.save(subgraph_list, subgraph_list_path)
        else:
            print('Subgraph cache found, skip rebuilding.')
            subgraph_list = []

        return kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index

    def load_align_links(self):
        """Load seed alignment pairs; returns {(lang1, lang2): LongTensor}."""
        seeds = {}
        for f in os.listdir(self.align_dir):
            lang1 = f[:2]
            lang2 = f[3:5]
            links = pd.read_csv(
                os.path.join(self.align_dir, f), sep='\t', header=None,
            ).values.astype(int)
            links = torch.unique(torch.LongTensor(links), dim=0)
            seeds[(lang1, lang2)] = links
        return seeds

    def load_kg_data(self, language):
        """Load train/val/test triples and entity/relation counts for one domain."""
        train_df = pd.read_csv(
            os.path.join(self.kg_dir, language + '-train.tsv'),
            sep='\t', header=None, names=['head', 'relation', 'tail'],
        )[:self.MAX_SAM]
        val_df = pd.read_csv(
            os.path.join(self.kg_dir, language + '-val.tsv'),
            sep='\t', header=None, names=['head', 'relation', 'tail'],
        )[:self.MAX_SAM]
        test_df = pd.read_csv(
            os.path.join(self.kg_dir, language + '-test.tsv'),
            sep='\t', header=None, names=['head', 'relation', 'tail'],
        )[:self.MAX_SAM]

        print('load_kg_data', len(train_df))

        with open(os.path.join(self.entity_dir, language + '.tsv'), encoding='utf-8') as entity_file:
            num_entities = len(entity_file.readlines())

        with open(os.path.join(self.data_dir, 'relations.txt')) as relation_file:
            num_relations = len(relation_file.readlines())

        return (
            torch.LongTensor(train_df.values.astype(int)),
            torch.LongTensor(val_df.values.astype(int)),
            torch.LongTensor(test_df.values.astype(int)),
            num_entities,
            num_relations,
        )
