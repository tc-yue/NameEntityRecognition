# -*- coding: utf-8 -*-
# @Time    : 2019/6/15 18:38
# @Author  : Tianchiyue
# @File    : data_process.py.py
# @Software: PyCharm

from data_utils import *
from utils import *
from pytorch_pretrained_bert.tokenization import BertTokenizer


def bert_process(sentences, path='bert-base-chinese'):
    tokenizer = BertTokenizer.from_pretrained(path)
    id_list = []
    for tokens in sentences:
        ids = [tokenizer.vocab[token] if token in tokenizer.vocab else tokenizer.vocab['[UNK]'] for token in
               ["[CLS]"] + tokens]
        id_list.append(ids)
    return id_list


def read_corpus(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset) as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                word, tag = line.split('\t')
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


if __name__ == '__main__':
    if not os.path.exists('processed'):
        os.mkdir('processed/')
    mode = 'nn'
    train_data = read_corpus('data/msra_train_bio')
    test_data = read_corpus('data/msra_test_bio')
    all_text = [i[0] for i in train_data + test_data]

    max_length = 150

    all_label = [i[1] for i in train_data + test_data]
    tagset = list(set([item for sublist in all_label for item in sublist]))
    tag2index = {}
    for tag in tagset:
        if tag2index.get(tag) is None:
            tag2index[tag] = len(tag2index)
    index2tag = {v: k for k, v in tag2index.items()}
    index = {}
    if mode == 'bert':
        all_x = bert_process(all_text)
        all_y = [[-1] + list(map(lambda w: tag2index[w], sublist)) for sublist in all_label]
        index['word2id'] = [None]
    else:
        tokenizer = Tokenizer(oov_token='<UNK>')
        tokenizer.fit_on_texts(all_text)
        all_x = tokenizer.texts_to_sequences(all_text)
        all_y = [list(map(lambda w: tag2index[w], sublist)) for sublist in all_label]
        index['word2id'] = tokenizer.word_index
    train_x_pad = pad_sequences(all_x[:len(train_data)], maxlen=max_length, padding='post')
    test_x_pad = pad_sequences(all_x[len(train_data):], maxlen=max_length, padding='post')
    train_y_pad = pad_sequences(all_y[:len(train_data)], maxlen=max_length, padding='post', value=-1)
    test_y_pad = pad_sequences(all_y[len(train_data):], maxlen=max_length, padding='post', value=-1)
    write_pickle('processed/{}_msra_train_data.pkl'.format(mode), (train_x_pad, train_y_pad))
    write_pickle('processed/{}_msra_test_data.pkl'.format(mode), (test_x_pad, test_y_pad))

    index['tag2id'] = tag2index
    write_pickle('processed/{}_msra_index.pkl'.format(mode), index)
