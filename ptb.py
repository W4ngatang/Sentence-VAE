import os
import io
import json
import ipdb as pdb
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import TweetTokenizer

from utils import OrderedCounter

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super(PTB, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 4)
        self.max_vocab_size = kwargs.get('max_vocab_size', 30000)

        self.raw_data_file = os.path.join(data_dir, split+'.txt')
        self.data_file = os.path.join(data_dir, split + '.json')
        self.vocab_file = os.path.join(data_dir, 'vocab.json')

        if create_data:
            print("Creating new %s data." % split.upper())
            self._create_data()

        elif not os.path.exists(self.data_file):
            print("%s preprocessed file not found at %s. Creating new." % (split.upper(), self.data_file))
            self._create_data()

        else:
            print("Loading data from %s." % self.data_file)
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(self.data_file, 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(self.vocab_file, 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(self.vocab_file, 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)
        with open(self.raw_data_file, 'r') as file:

            for i, line in enumerate(file):

                words = nltk.word_tokenize(line)

                #input = ['<sos>'] + words
                #input = input[:self.max_sequence_length]
                input = words[:self.max_sequence_length]

                #target = words[:self.max_sequence_length-1]
                #target = target + ['<eos>']
                target = words[:self.max_sequence_length]

                assert len(input) == len(target), "%i, %i" % (len(input), len(target))
                length = len(input)

                #input.extend(['<pad>'] * (self.max_sequence_length-length))
                #target.extend(['<pad>'] * (self.max_sequence_length-length))

                #input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                #target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(self.data_file, 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_file, 'r') as file:

            for i, line in enumerate(file):
                words = nltk.word_tokenize(line)
                w2c.update(words)

            word_freq_pairs = [(word, freq) for word, freq in w2c.items()]
            word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
            for word, freq in word_freq_pairs[:self.max_vocab_size - len(special_tokens)]:
                i2w[len(w2i)] = word
                w2i[word] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(self.vocab_file, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
