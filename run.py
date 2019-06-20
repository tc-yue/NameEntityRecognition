# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 12:46
# @Author  : Tianchiyue
# @File    : run.py
# @Software: PyCharm

import argparse
import torch.nn.functional as F
from utils import *
import json
import os
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, BertForMaskedLM, BertModel, \
    BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from model import NERModel
from bertmodel import NERBert
from metric import f1_score

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--seed_num', default=147, type=int)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--valid_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--index_path', type=str, default=None)

    parser.add_argument('--bert_config_path', type=str, default='bert_chinese/bert_config.json')
    parser.add_argument('--bert_model_path', type=str, default='bert_chinese/pytorch_model.bin')

    parser.add_argument('--model', default='cnn', type=str)
    parser.add_argument('-e', '--epochs', default=10, type=int)

    parser.add_argument('-d', '--dropout', default=0.2, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--train_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--valid_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')

    parser.add_argument('--gradient_accumulation_steps',default=1,type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--valid_step', default=1000, type=int)

    parser.add_argument('--vocab_size', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_labels', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--embedding_dim', type=int, default=300, help='DO NOT MANUALLY SET')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--trainable_embedding', action='store_true')
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument('--weighted', action='store_true')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def get_dataloader(train_data_path, batch_size, shuffle):
    train_x, train_y = read_pickle(train_data_path)
    num_sample = len(train_x)
    train_x = torch.tensor(train_x, dtype=torch.long)
    train_y = torch.tensor(train_y, dtype=torch.long)
    train_data = TensorDataset(train_x, train_y)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return dataloader, num_sample


def train_eval(args, train_data_path, valid_data_path):

    index = read_pickle(args.index_path)
    word2index, tag2index = index['word2id'], index['tag2id']
    args.num_labels = len(tag2index)
    args.vocab_size = len(word2index)+1
    set_seed(args.seed_num)
    train_dataloader, train_samples = get_dataloader(train_data_path, args.train_batch_size, True)
    valid_dataloader, _ = get_dataloader(valid_data_path, args.valid_batch_size, False)

    if args.model == 'bert':
        bert_config = BertConfig(args.bert_config_path)
        model = NERBert(bert_config, args)
        model.load_state_dict(torch.load(args.bert_model_path), strict=False)
        # model = NERBert.from_pretrained('bert_chinese',
        #                                 # cache_dir='/home/dutir/yuetianchi/.pytorch_pretrained_bert',
        #                                 num_labels=args.num_labels)
    else:
        if args.embedding:
            word_embedding_matrix = read_pickle(args.embedding_data_path)
            model = NERModel(args, word_embedding_matrix)
        else:
            model = NERModel(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.model == 'bert':
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if 'bert' not in n], 'lr': 5e-5, 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and ('bert' in n)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and ('bert' in n)],
             'weight_decay': 0.0}
        ]
        warmup_proportion = 0.1
        num_train_optimization_steps = int(
            train_samples / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    else:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    global_step = init_step
    best_score = 0.0

    logging.info('Start Training...')
    logging.info('init_step = %d' % global_step)
    for epoch_id in range(int(args.epochs)):

        tr_loss = 0
        model.train()
        for step, train_batch in enumerate(train_dataloader):


            batch = tuple(t.to(device) for t in train_batch)
            _, loss = model(batch[0], batch[1])
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % 500 == 0:
                print(loss.item())

            if args.do_valid and global_step % args.valid_step == 1:
                true_res = []
                pred_res = []
                len_res = []
                model.eval()
                for valid_step, valid_batch in enumerate(valid_dataloader):
                    valid_batch = tuple(t.to(device) for t in valid_batch)

                    with torch.no_grad():
                        logit = model(valid_batch[0])
                    if args.model == 'bert':
                        # 第一个token是‘cls’
                        len_res.extend(torch.sum(valid_batch[0].gt(0), dim=-1).detach().cpu().numpy()-1)
                        true_res.extend(valid_batch[1].detach().cpu().numpy()[:,1:])
                        pred_res.extend(logit.detach().cpu().numpy()[:,1:])
                    else:
                        len_res.extend(torch.sum(valid_batch[0].gt(0),dim=-1).detach().cpu().numpy())
                        true_res.extend(valid_batch[1].detach().cpu().numpy())
                        pred_res.extend(logit.detach().cpu().numpy())
                acc, score = cal_score(true_res, pred_res, len_res, tag2index)
                score = f1_score(true_res, pred_res, len_res, tag2index)
                logging.info('Evaluation:step:{},acc:{},fscore:{}'.format(str(epoch_id), acc, score))
                if score>=best_score:
                    best_score = score
                    if args.model == 'bert':
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_dir = '{}_{}'.format('bert', str(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            output_config_file = os.path.join(output_dir, CONFIG_NAME)
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())
                    else:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': args.learning_rate,
                            'warm_up_steps': step
                        }
                        save_model(model, optimizer, save_variable_list, args)
                model.train()


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    train_eval(args, args.train_data_path, args.valid_data_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
