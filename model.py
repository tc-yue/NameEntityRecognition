# -*- coding: utf-8 -*-
# @Time    : 2019/6/15 17:11
# @Author  : Tianchiyue
# @File    : model.py
# @Software: PyCharm

import torch.nn as nn
import torch
from crf import ConditionalRandomField
import torch.nn.functional as F
from layers import DynamicRNN

class CNNEncoder(nn.Module):

    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(args.embedding_dim, args.hidden_dim, 5, padding=2)
        self.conv2 = nn.Conv1d(args.embedding_dim, args.hidden_dim, 3, padding=1)
        self.dropout = nn.Dropout(args.dropout)
        self.conv3 = nn.Conv1d(args.hidden_dim, args.hidden_dim, 5, padding=2)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        x_conv = F.relu(torch.cat([self.conv1(inputs), self.conv2(inputs)], dim=1), inplace=True)
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.conv3(x_conv), inplace=True)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return x_conv


class NERModel(nn.Module):

    def __init__(self, args, word_emb_matrix=None):
        super(NERModel, self).__init__()
        self.args = args

        if word_emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_emb_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = args.trainable_embedding
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
            self.embedding.weight.requires_grad = True
        if args.model == 'cnn':
            self.encoder = CNNEncoder(args)
        elif args.model == 'rnn':
            self.encoder = DynamicRNN(args.embedding_dim, args.hidden_dim, bidirectional=True)

        self.linear = nn.Linear(args.hidden_dim*2, args.num_labels)
        self.dropout = nn.Dropout(0.2)
        if self.args.use_crf:
            self.crf = ConditionalRandomField(args.num_labels)

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.gt(0)
        inputs = self.embedding(input_ids)
        if self.args.model == 'cnn':

            rep = self.encoder(inputs)
        elif self.args.model == 'rnn':
            x_len = torch.sum(input_ids != 0, dim=1)
            rep, _ = self.encoder(inputs, x_len)
        logits = self.linear(self.dropout(rep))

        if not self.args.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                # Only keep active parts of the loss
                loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))
                return logits, loss
            else:
                return logits
        else:
            if labels is not None:
                # Only keep active parts of the loss
                if attention_mask is not None:
                    total_loss = self.crf(logits, labels, attention_mask)
                    return 0,torch.mean(total_loss)
            else:
                max_len = logits.shape[1]

                tag_seq = self.crf.viterbi_decode(logits, attention_mask)
                for pred in tag_seq:
                    if len(pred) < max_len:
                        pred += [0] * (max_len - len(pred))
                return tag_seq