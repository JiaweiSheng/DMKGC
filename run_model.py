#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join

print('Current working dir', os.getcwd())

import numpy as np
import torch
from src.data_loader import MKGDataset
from src.validate import Tester
from src.utils import (
    nodes_to_k_graph,
    get_k_subgraph_list,
    get_negative_samples_graph,
    save_model,
)

import logging
import argparse
import random
from random import SystemRandom
import time
from itertools import cycle

from transformers import AdamW, get_linear_schedule_with_warmup


def set_logger(model_dir, args):
    """Write logs to checkpoint directory and console."""
    experimentID = int(SystemRandom().random() * 100000)
    log_file = model_dir + "/" + '_'.join(args.langs) + "_train_" + args.model +'_'+args.v+ '_' +str(experimentID) + ".log"
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return experimentID


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models'
    )

    # Data
    parser.add_argument('--remove_language', type=str, default='', help='comma-separated domain codes to exclude')
    parser.add_argument('--k', default=10, type=int, help='number of neighbor nodes to sample')
    parser.add_argument('--num_hops', default=2, type=int, help='subgraph hop count')
    parser.add_argument('--data_path', default='dataset', type=str, help='data root directory')
    parser.add_argument('--dataset', default='dbp5l', type=str, help='dataset name')
    parser.add_argument('--save', default='T', type=str, help='save best checkpoint when T')
    parser.add_argument('--MAX_SAM', default=10000000000, type=int, help='max samples for debugging')

    # Model
    parser.add_argument('--model', default='dmkgc', type=str, help='model name')
    parser.add_argument('--margin', default=0.5, type=float, help='TransE margin')
    parser.add_argument('--dim', default=256, type=int, help='entity/relation embedding dimension')
    parser.add_argument('--n_layers_gnn', default=2, type=int, help='GNN layer count')
    parser.add_argument('--encoder_hdim_gnn', default=256, type=int, help='GNN hidden dimension')
    parser.add_argument('--n_heads', default=1, type=int, help='attention head count')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--v_att', default='cxt', type=str, help='attention fusion variant: cxt, cxt+cur, cxt+mean')

    # Diffusion
    parser.add_argument('--n_steps', default=50, type=int, help='diffusion training steps')
    parser.add_argument('--n_sampling_step', default=50, type=int, help='diffusion sampling steps')
    parser.add_argument('--beta_start', default=0.0001, type=float, help='beta schedule start')
    parser.add_argument('--beta_end', default=0.02, type=float, help='beta schedule end')
    parser.add_argument('--beta_sche', default='exp', type=str, help='beta schedule: linear, exp, cosine, sqrt')
    parser.add_argument('--diffuser_type', default='mlp2', type=str, help='denoiser MLP type: mlp1, mlp2')
    parser.add_argument('--p_uncond', default=0.1, type=float, help='classifier-free guidance dropout probability')
    parser.add_argument('--s_strength', default=2, type=float, help='CFG sampling strength')
    parser.add_argument('--w_recon', default=0.01, type=float, help='reconstruction loss weight')
    parser.add_argument('--w_reg', default=0.001, type=float, help='cross-domain regularization loss weight')

    # Training
    parser.add_argument('--epoch_each', default=1, type=int, help='epochs per round')
    parser.add_argument('--round', default=30, type=int, help='training rounds')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=300, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='evaluation batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or AdamW')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--scheduler', type=str, default='linear', help='constant or linear')
    parser.add_argument('--warmup', default=1, type=int, help='warmup rounds for linear scheduler')
    parser.add_argument('--lw', type=str, default='n', choices=['y', 'n'], help='reweight loss by domain batch count')

    # Runtime
    parser.add_argument('--device', default='cuda:0', type=str, help='torch device')
    parser.add_argument('--v', default='', type=str, help='experiment tag for log/checkpoint naming')

    return parser.parse_args(args)


def train_kgs(args, all_langs, kg_objects_dict, kgname2idx, optimizer, num_epoch, model, all_entity_global_index, scheduler=None):
    max_data = 0
    kg_dataloader_list = {}
    lang_nbatch = {}
    for lang in all_langs:
        kg = kg_objects_dict[lang]
        kg_index = kgname2idx[lang]
        kg_dataloader = kg.generate_batch_data(
            kg.h_train, kg.r_train, kg.t_train,
            batch_size=args.batch_size, shuffle=True,
        )
        kg_dataloader_list[lang] = cycle(kg_dataloader)
        max_data = max(max_data, len(kg_dataloader))
        lang_nbatch[lang] = len(kg_dataloader)
        logging.info('Domain {}: nbatch {}'.format(lang, len(kg_dataloader)))
    logging.info('Largest data has {} batches'.format(max_data))

    if args.lw == 'y':
        lang_lw = {}
        for lang in all_langs:
            lang_lw[lang] = lang_nbatch[lang] / max_data
            logging.info('{}, {}, {}, {}'.format(lang, lang_nbatch[lang], max_data, lang_lw[lang]))

    for one_epoch in range(num_epoch):
        logging.info('Epoch {:d}'.format(one_epoch))
        kg_loss = []
        all_loss = []

        for i in range(max_data):
            random.shuffle(all_langs)
            for lang in all_langs:
                time0 = time.time()
                kg_dataloader = kg_dataloader_list[lang]
                kg_each = next(kg_dataloader)

                kg = kg_objects_dict[lang]
                kg_index = kgname2idx[lang]

                h_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, kg_each[:, 0], args.device)
                t_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, kg_each[:, 2], args.device)

                batch_size = kg_each.shape[0]
                t_neg_index = get_negative_samples_graph(batch_size, kg.num_entities)
                t_neg_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, t_neg_index, args.device)

                kg_each = kg_each.to(args.device)
                t_neg_index = t_neg_index.to(args.device)
                optimizer.zero_grad()
                total_loss = model.forward_kg(
                    h_graph_batch_list, kg_each, t_graph_batch_list,
                    t_neg_graph_batch_list, kg_index, t_neg_index,
                )

                loss = (
                    total_loss['kg_loss']
                    + total_loss['loss_recon'] * args.w_recon
                    + total_loss['loss_reg'] * args.w_reg
                )

                if args.lw == 'y':
                    loss = loss * lang_lw[lang]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                all_loss.append(loss.item())
                kg_loss.append(total_loss['kg_loss'].item())

                if i % 10 == 0:
                    logging.info(
                        'Step {}: Domain: {}, Train KG Loss: all_loss {:.6f}, kg_loss {:.6f}, '
                        'loss_recon {:.6f}, kg_loss_reg {:.6f}'.format(
                            i, lang, loss.item(), total_loss['kg_loss'].item(),
                            total_loss['loss_recon'].item(), total_loss['loss_reg'].item(),
                        )
                    )

                del loss
                torch.cuda.empty_cache()

                print('time each batch:', time.time() - time0)
        logging.info(
            'Epoch {:d} [Train KG Loss: all_loss {:.6f}, kg_loss {:.6f}]'.format(
                one_epoch, np.mean(all_loss), np.mean(kg_loss),
            )
        )
        logging.info('\n')


def main(args):
    args.device = torch.device(args.device)
    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim

    if args.dataset == 'dbp5l':
        all_langs = ['el', 'en', 'es', 'fr', 'ja']
    elif args.dataset == 'depkg':
        all_langs = ['de', 'es', 'fr', 'it', 'jp', 'uk']
    elif args.dataset == 'dwy':
        all_langs = ['db', 'wk', 'yg']
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    remove_lang = args.remove_language
    if remove_lang != '':
        remove_lang = remove_lang.split(',')
        for la in remove_lang:
            all_langs.remove(la)

    print(f"Number of KGs is {len(all_langs)}")
    args.langs = all_langs
    kgname2idx = {}
    for i in range(len(all_langs)):
        kgname2idx[all_langs[i]] = i
    all_langs_out = list.copy(all_langs)

    model_dir = join('./' + args.dataset + "/trained_model", '_'.join(all_langs))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    experimentID = set_logger(model_dir, args)
    logging.info('logger setting finished')

    dataset = MKGDataset(args, MAX_SAM=args.MAX_SAM)
    kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index = dataset.load_data()
    logging.info('subgraph_list loaded')

    # Map each global entity to the domain KGs it appears in
    all_entity2kgidx = {}
    for global_id, langs in all_entity2kgs.items():
        langidx = set([kgname2idx[l] for l in langs])
        all_entity2kgidx[global_id] = langidx

    graph_dir = '_'.join(args.langs)
    for lang in kg_objects_dict.keys():
        kg_lang = kg_objects_dict[lang]
        kg_index = kgname2idx[lang]
        node_index = kg_lang.entity_global_index

        k_subgraph_list = get_k_subgraph_list(
            subgraph_list, node_index, kg_index, dataset.num_kgs,
            all_entity2kgidx, dataset.num_entities,
            os.path.join(dataset.data_dir, graph_dir),
        )
        logging.info('kg' + str(kg_index) + '_k_subgraph_list loaded')
        kg_lang.k_subgraph_list = k_subgraph_list

    args.num_entities = dataset.num_entities
    args.num_relations = dataset.num_relations
    args.num_kgs = dataset.num_kgs
    args.kgname2idx = kgname2idx

    del subgraph_list

    logging.info('remove domain: %s' % (remove_lang))
    logging.info(f'domains: {args.langs}')
    logging.info(f'device: {args.device}')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'k: {args.k}')
    logging.info(f'num_hops: {args.num_hops}')
    logging.info(f'lr: {args.lr}')
    logging.info(f'margin: {args.margin}')
    logging.info(f'dim: {args.dim}')
    logging.info(f'experimentID: {experimentID}')
    logging.info(f'MAX_SAM: {args.MAX_SAM}')

    if args.model == 'dmkgc':
        from src.dmkgc import DMKGC
        model = DMKGC(args).to(args.device)
    else:
        assert True, 'unimplemented'

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_steps = 0
    for lang in kg_objects_dict.keys():
        kg_lang = kg_objects_dict[lang]
        kg_dataloader = kg_lang.generate_batch_data(
            kg_lang.h_train_global, kg_lang.r_train, kg_lang.t_train_global,
            batch_size=args.batch_size, shuffle=True,
        )
        num_steps += len(kg_dataloader)
    args.num_steps = num_steps

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.epoch_each * args.num_steps * args.warmup,
            num_training_steps=args.round * args.epoch_each * args.num_steps,
        )
    else:
        raise NotImplementedError

    logging.info('model initialization done')

    validator = Tester(args, kg_objects_dict, model, args.device, args.data_path + args.dataset)

    best_mrr = 0
    best_result = {}

    logging.info(f'=== experimentID {experimentID} ===')
    for i in range(args.round):
        logging.info(f'Round: {i} begin!')
        model.train()

        train_kgs(
            args, all_langs, kg_objects_dict, kgname2idx, optimizer,
            args.epoch_each, model, scheduler=scheduler,
            all_entity_global_index=all_entity_global_index,
        )
        logging.info(f'round : {i} finished!')

        model.eval()
        with torch.no_grad():
            metrics_test2 = validator.test(is_val=False, is_filtered=True)

            filename = "experiment_" + str(experimentID) + '_best.ckpt'
            mean_mrr = np.mean([metrics_test2[lang][2].item() for lang in all_langs])
            logging.info(f'cur epoch: {i}, cur mean mrr: {mean_mrr}!')
            logging.info(f'Round {i} finished!')

            if best_mrr < mean_mrr:
                best_mrr = mean_mrr
                best_epoch = i
                best_result = metrics_test2
                best_filename = filename
                if args.save == 'T':
                    save_model(model, model_dir, best_filename, args)

            logging.info(f'best epoch: {best_epoch}, best mean mrr: {best_mrr}!')
            for lang in all_langs_out:
                logging.info(
                    '{} filterd: {:.4f}, {:.4f}, {:.4f}'.format(
                        lang, best_result[lang][0], best_result[lang][1], best_result[lang][2],
                    )
                )


if __name__ == "__main__":
    main(parse_args())
