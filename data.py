from torchtext.data import TabularDataset
from torch.utils.data import Dataset , DataLoader
from datasets import load_dataset
import random
import re
import os
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data
from nltk.tokenize import word_tokenize

import spacy
from nltk import tokenize
from torchtext.data.utils import get_tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

class wmtloader:

    def __init__(self,train_path , valid_path,max_length = 255,
                 batch_size = 16,
                 device='cpu'):

        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            lower=True,
#            init_token='<BOS>',
#            eos_token='<EOS>',
            tokenize=de_tokenizer
        )
        self.tgt = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            init_token='<BOS>',
            eos_token='<EOS>',
            tokenize=en_tokenizer
        )

        train_data = TabularDataset(
            path=train_path,
            format = 'csv',
            fields=[('de',self.src),('en',self.tgt)],
        )
        valid_data = TabularDataset(
            path=valid_path,
            format = 'csv',
            fields=[('de', self.src), ('en', self.tgt)],
        )
        self.train_iter = data.BucketIterator(
            train_data,
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=True,
            sort_key = lambda x: len(x.de),
            sort_within_batch=True,
        )
        self.valid_iter = data.BucketIterator(
            valid_data,
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=False,
            sort_key=lambda x: len(x.de),
            sort_within_batch=True,
        )

        self.src.build_vocab(train_data,max_size = 99999999)
        self.tgt.build_vocab(train_data,max_size = 99999999)
#
# wmt_loader = wmtloader('wmt16imsi.csv','wmt16_valid.csv',
#                         batch_size=16,device=0)
#
#
# for i , train_data in enumerate(wmt_loader.train_iter):
#      print(train_data)
#      pass
#
#






