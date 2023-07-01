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
import csv
from eunjeon import Mecab
import spacy
from nltk import tokenize
import sentencepiece as sim

de_tokenizer = spacy.load("de_core_news_sm")
en_tokenizer = spacy.load("en_core_web_sm")

# de_processor ,en_processor= sim.SentencePieceProcessor() ,  sim.SentencePieceProcessor()
# de_vocab_model , en_vocab_model = 'dataset/wmt16_train200_de.model' , 'dataset/wmt16_train200_en.model'
# de_processor.load(de_vocab_model)
# en_processor.load(en_vocab_model)
#
# from torchtext.data.utils import get_tokenizer
# en_tokenizer = de_processor.encode_as_pieces
# de_tokenizer = en_processor.encode_as_pieces


class wmtloader:

    def __init__(self,train_path , valid_path,max_length = 256,
                 batch_size = 16,
                 device='cpu',fixed_length = True):

        super(wmtloader, self).__init__()

        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
#            include_lengths=True,
           lower=True,
#            init_token='<BOS>',
#            eos_token='<EOS>',
#            tokenize=de_tokenizer
        )
        self.trg = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
#            include_lengths=True,
            init_token='<BOS>',
            eos_token='<EOS>',
           # tokenize=en_tokenizer
        )
        if fixed_length:
            train_data = TranslationDataset_separated(    #tsv
                path=train_path,
                fields=[('src',self.src),('trg',self.trg)],
                max_length = max_length-2
            )
            valid_data = TranslationDataset_separated(
                path=valid_path,
                fields=[('src', self.src), ('trg', self.trg)],
                max_length=max_length-2
            )
        else:
            train_data = TabularDataset(
                path=train_path,
                format='csv',
                fields=[('src', self.src), ('trg', self.trg)],
            )
            valid_data = TabularDataset(
                path=valid_path,
                format='csv',
                fields=[('src', self.src), ('trg', self.trg)],
            )

        self.train_iter = data.BucketIterator(
            train_data,
            batch_size=batch_size,
            device='cpu',
            shuffle=True,
           sort_key = lambda x: len(x.src) + (max_length * len(x.trg)),
           sort_within_batch=True,
        )
        self.valid_iter = data.BucketIterator(
            valid_data,
            batch_size=batch_size,
            device='cpu',
            shuffle=False,
           sort_key= lambda x: len(x.src) + (max_length * len(x.trg)),
           sort_within_batch=True,
        )

        self.src.build_vocab(train_data,max_size = 99999999)
        self.trg.build_vocab(train_data,max_size = 99999999)



class TranslationDataset_separated(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path , fields, max_length=None, **kwargs):

        src_tgt = ['.de','.en']
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path , trg_path =tuple(os.path.expanduser(path + x) for x in src_tgt)

        self.examples = []
        with open(src_path, encoding='utf-8') as src_file , open(trg_path,encoding='utf-8') as trg_file:
            for src_line , trg_line in zip(src_file,trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length <  max(len(de_tokenizer(src_line)), len(en_tokenizer(trg_line))):
                    continue
                if src_line != '' and trg_line != '':
                    self.examples += [data.Example.fromlist([src_line, trg_line], fields)]

        super().__init__(self.examples, fields, **kwargs)



class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path , fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        with open(path, encoding='utf-8') as data_file:
            for line in csv.reader(data_file,delimiter='\t'):
                # if len(line.split('\t') != 2):
                #     continue
                src_line, trg_line = line
                src_line , trg_line = src_line.strip('\n') , trg_line.strip('\n')
                if max_length and max_length < max(len(de_tokenizer(src_line)), len(en_tokenizer(trg_line))):
                    continue
                if src_line != '' and trg_line != '':
                    examples += [data.Example.fromlist([src_line, trg_line], fields)]

        super().__init__(examples, fields, **kwargs)



class koenloader:

    def __init__(self,train_path , valid_path,max_length = 256,
                 batch_size = 16,
                 device='cpu'):

        mecab = Mecab()
        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
#            include_lengths=True,
            lower=True,
#            init_token='<BOS>',
#            eos_token='<EOS>',
            tokenize=mecab.morphs
        )
        self.trg = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
#            include_lengths=True,
            init_token='<BOS>',
            eos_token='<EOS>',
            tokenize=en_tokenizer
        )

        train_data = TabularDataset(
            path=train_path,
            format = 'tsv',
            fields=[('src',self.src),('trg',self.trg)],
        )
        valid_data = TabularDataset(
            path=valid_path,
            format = 'tsv',
            fields=[('src', self.src), ('trg', self.trg)],
        )
        self.train_iter = data.BucketIterator(
            train_data,
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=True,
           sort_key = lambda x: len(x.src) + (max_length * len(x.trg)),
           sort_within_batch=True,
        )
        self.valid_iter = data.BucketIterator(
            valid_data,
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=False,
           sort_key= lambda x: len(x.src) + (max_length * len(x.trg)),
           sort_within_batch=True,
        )

        self.src.build_vocab(train_data,max_size = 99999999)
        self.trg.build_vocab(train_data,max_size = 99999999)

# wmt_loader = wmtloader('dataset/wmt16_train_10m','dataset/wmt16_valid',max_length=64, batch_size=16,device=-1)
# # model = transformer.Transformer(len(wmt_loader.src.vocab),len(wmt_loader.trg.vocab),
# #                                         layers=4,
# #                                         hidden_size=768,
# #                                         heads=8,
# #                                         dropout_p=0.1,
# #                                         max_length=64)
# # wmt_loader = koenloader('dataset/corpus.shuf.train_test.tsv','dataset/corpus.shuf.valid_test.tsv', batch_size=16,device=0)
# #
# for i , train_data in enumerate(wmt_loader.train_iter):
#      print(train_data)
#      pass
#







