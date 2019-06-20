# -*- coding: utf-8 -*-
# @Time    : 2019/6/15 17:11
# @Author  : Tianchiyue
# @File    : bertmodel.py
# @Software: PyCharm
from crf import ConditionalRandomField
from pytorch_pretrained_bert.modeling import BertForTokenClassification,BertForMaskedLM,BertModel,BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class NERBert(BertPreTrainedModel):
    def __init__(self, config, args):
        super(NERBert, self).__init__(config)
        self.args = args
        self.bert = BertModel(config)
        if self.args.weighted:
            self.weight = nn.Parameter(torch.Tensor(self.args.num_layers))
            self.weight.data.uniform_(0.1, 0.9)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.args.num_labels)
        if self.args.use_crf:
            self.crf = ConditionalRandomField(self.args.num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.gt(0)
        encoded_layers, _ = self.bert(input_ids, None, attention_mask, output_all_encoded_layers=True)
        if not self.args.weighted:
            sequence_output = encoded_layers[-1]
        else:
            last_layers = torch.cat(encoded_layers[-self.num_layers:], dim=-1).view(encoded_layers[0].size(0),
                                                                                    encoded_layers[0].size(1),
                                                                                    encoded_layers[0].size(2),
                                                                                    self.num_layers)
            soft_weight = F.softmax(self.weight)
            sequence_output = torch.matmul(last_layers, soft_weight)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
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
                    return torch.mean(total_loss)
            else:
                max_len = logits.shape[1]

                tag_seq = self.crf.viterbi_decode(logits, attention_mask)
                for pred in tag_seq:
                    if len(pred) < max_len:
                        pred += [0] * (max_len - len(pred))
                return tag_seq