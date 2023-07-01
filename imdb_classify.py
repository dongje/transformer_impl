


import argparse
import torch
import time
from torchtext.data import TabularDataset
import torch.nn as nn
import idbm_dataloader
import transformer
from torch import optim
from tqdm import tqdm
from torch.cuda.amp import autocast , GradScaler
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def define_arguparser():

    p = argparse.ArgumentParser()

    p.add_argument('--load_model_name' , required=True)
    p.add_argument('--test_data_path',required=True)
    p.add_argument('--device',required=True,type=int ,default=-1)
    p.add_argument('--batch_size',type=int , default = 64)
    p.add_argument('--max_length',type=int , default=256)

    config = p.parse_args()

    return config


def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields.
    With those fields, we can retore mapping table between words and indice.
    '''
    return (
        data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            #            include_lengths=True,
            lower=True,
            init_token='<BOS>',
            #            eos_token='<EOS>',
            tokenize=en_tokenizer
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
        )
    )

def test_load(config,text_field,label_field):

    test_data = TabularDataset(
        path=config.test_data_path,
        format='csv',
        fields=[('review', text_field), ('sentiment', label_field)],
    )
    test_iter = data.BucketIterator(
        test_data,
        batch_size=config.batch_size,
        device='cuda:%d' % config.device if config.device >= 0 else 'cpu',
#        shuffle=True,
        sort_key=lambda x: len(x.review),
        sort_within_batch=True,
    )
    return test_iter


def getacc(y_hat,y):
    y_result = torch.argmax(y_hat,dim=-1)
    return (y_result == y.contiguous().view(-1)).sum() / y_result.size(0)

def test(model,test_loader , crit,config):

    device = next(model.parameters()).device
    print(device)
    start_time = time.time()
    total_acc = 0
    total_loss = 0
    for i , mini_batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            mini_batch.review = mini_batch.review.to(device)
            mini_batch.sentiment = mini_batch.sentiment.to(device)

            x, y = mini_batch.review, mini_batch.sentiment
            x = x[:, : config.max_length]

            y_hat = model(x)  # y_hat = |batch_size , output_size| (softmaxed)
            #                print(y_hat)

            loss = crit(
                y_hat.contiguous(),  # | bs  , output_size|
                y.contiguous().view(-1)  # |bs|
            )
            #                backward_target = loss.div(y_hat.size(0)).div(config.iteration_per_update)
            backward_target = loss

            acc = getacc(y_hat, y)
            total_acc += acc
            total_loss += loss
            print(f'iter : {i} loss : {loss} acc : {acc}')
    total_loss /= len(test_loader)
    total_acc /= len(test_loader)
    end_time = time.time()
    print(f'total loss : {total_loss} , total acc : {total_acc}')
    print(f'test time : {end_time - start_time} seconds')
    return total_acc


def main(config):

    saved_data = torch.load(
        config.load_model_name,
        map_location='cpu' if config.device < 0 else 'cuda:%d' % config.device
    )

    train_config = saved_data['config']
    saved_model = saved_data['model']
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']
    model = transformer.encoder_classifier(len(src_vocab),len(tgt_vocab),layers=train_config.layers,hidden_size=train_config.hidden_size , heads=train_config.heads
                                           , dropout_p=train_config.dropout_p , max_length=train_config.max_length,first_norm=train_config.first_norm)
    model.load_state_dict(saved_model)


    text_field, label_field = define_field()
    text_field.vocab = src_vocab
    label_field.vocab = tgt_vocab

    test_loader = test_load(config,text_field,label_field)
    crit = nn.NLLLoss(reduction='mean')

    if config.device >=0:
        model.cuda(config.device)
        crit.cuda(config.device)

    acc_list = []
    m = 10
    for i in range(m):
        acc_list.append(test(model,test_loader,crit,config))
    print(f'\n 평균 {m} accuracy : {sum(acc_list)/ m }')




if __name__ == '__main__':
    config = define_arguparser()
    main(config)


