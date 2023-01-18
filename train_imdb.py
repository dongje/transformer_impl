


import argparse
import torch
import torch.nn as nn
import idbm_dataloader
import transformer
from torch import optim
from tqdm import tqdm
from torch.cuda.amp import autocast , GradScaler
import numpy as np
import torch_optimizer as custom_optim
from torch.optim import Optimizer
import transformer_copy
import transformer_github


## true , 'copy' , 'torch' , 'github
#original = 'github'
original = True

def define_arguparser():

    p = argparse.ArgumentParser()
    is_continue = False

    p.add_argument('--load_name' , required=is_continue)
    p.add_argument('--init_epoch' , required=is_continue,type=int , default=1)

    p.add_argument('--model_name' , required=not is_continue)
    p.add_argument('--device',required=True,type=int )
    p.add_argument('--batch_size',type=int , default = 64)
    p.add_argument('--iteration_per_update',type=int,default=4)
    p.add_argument('--autocast',type=int , default=1)
    p.add_argument('--first_norm',type=bool,default=True)
    p.add_argument('--n_epochs', type=int,default=15)
    p.add_argument('--max_length',type=int , default=256)
    p.add_argument('--layers', type=int , default = 6)
    p.add_argument('--hidden_size' ,type=int , default = 768)
    p.add_argument('--heads' , type=int,default=8)
    p.add_argument('--dropout_p' ,type=float , default=0.2)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_freq', type=int, default=999999)

    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    config = p.parse_args()

    assert config.hidden_size % config.heads == 0

    return config


def get_crit(output_size,unk_index, pad_index):


    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    loss_weight = torch.ones_like(torch.FloatTensor([1] * output_size) )
    loss_weight[pad_index] = 0
    loss_weight[unk_index] = 0
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='mean'
    )
    return crit

def get_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98),eps=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    return optimizer


def save_model(totalacc,epoch, model,optimizer, config, src_vocab, tgt_vocab):

    # Set a filename for model of last epoch.
    # We need to put every information to filename, as much as possible.
    model_fn = config.model_name

    model_fn = [model_fn[:-1]] + ['%02d.' % epoch,
                                '%.3f' % totalacc
                                ] + [model_fn[-1]]

    model_fn = '.'.join(model_fn)

    # Unlike other tasks, we need to save current model, not best model.
    torch.save(
        {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'config': config,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
        }, model_fn
    )

def getacc(y_hat,y):
    y_result = torch.argmax(y_hat,dim=-1)
    return (y_result == y.contiguous().view(-1)).sum() / y_result.size(0)

def trainandvalidate(config,model,crit,optimizer,dataloader,scheduler = None):

    device =  next(model.parameters()).device
    if config.device >=0:
        scaler = GradScaler()

    for epoch in range(config.init_epoch,config.n_epochs):


        model.train()
        for i,mini_batch in tqdm(enumerate(dataloader.train_iter)):
            if i % config.iteration_per_update == 0:
                if i>0:
                    optimizer.zero_grad()
            mini_batch.review = mini_batch.review.to(device)
            mini_batch.sentiment = mini_batch.sentiment.to(device)
            with autocast(bool(config.autocast)):
                x,y = mini_batch.review , mini_batch.sentiment
                x = x[:, : config.max_length]

                if original == True:
                    y_hat = model(x)   # y_hat = |batch_size , output_size| (softmaxed)
                elif original =='github':
                    y_hat = model(x)
#                print(y_hat)

                loss = crit(
                    y_hat.contiguous(),    # | bs  , output_size|
                    y.contiguous().view(-1)                 # |bs|
                )
    #                backward_target = loss.div(y_hat.size(0)).div(config.iteration_per_update)
                backward_target=loss
#                print(torch.argmax(y_hat,dim=-1))
                if config.device >= 0 and bool(config.autocast):
                    scaler.scale(backward_target).backward()
                else:
                    backward_target.backward()

            if i % config.iteration_per_update == 0:
                if i>0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )
                    if config.device >= 0 and bool(config.autocast):
                    # Use scaler instead of optimizer.step() if using GPU.
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

            print(f'epoch : {epoch} , train iter : {i} loss : {loss} acc : {getacc(y_hat,y)}')

        #valid
        with torch.no_grad():
            model.eval()
            total_acc = 0
            for i, mini_batch in tqdm(enumerate(dataloader.valid_iter)):
                mini_batch.review = mini_batch.review.to(device)
                mini_batch.sentiment = mini_batch.sentiment.to(device)

                x, y = mini_batch.review, mini_batch.sentiment
                x = x[:, : config.max_length]
                with autocast(bool(config.autocast)):
                    if original == True:
                        y_hat = model(x)  # y_hat = |batch_size , output_size| (softmaxed)
                    elif original == 'github':
                        y_hat = model(x)
    #                print(y_hat)

                    loss = crit(
                        y_hat.contiguous(),  # | bs  , output_size|
                        y.contiguous().view(-1)  # |bs|
                    )
                #                backward_target = loss.div(y_hat.size(0)).div(config.iteration_per_update)
                backward_target = loss

                acc = getacc(y_hat,y)
                total_acc += acc
                print(f'epoch : {epoch} , valid iter : {i} loss : {loss} acc : {acc}')
            total_acc /= len(dataloader.valid_iter)
            print(f'epoch : {epoch} total acc : {total_acc}')
        save_model(total_acc,epoch,model,optimizer,config,
                   src_vocab=dataloader.src.vocab,
                   tgt_vocab=dataloader.tgt.vocab)



def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(                  #  decay_start lr부터 끝 epochs까지 계속 lr 감소.
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def main(config):

    print(config)

    imdb_loader = idbm_dataloader.imdb_loader('dataset/imdb_dataset_shuf_train.csv',
                        batch_size=config.batch_size,max_length=config.max_length,
                        device=config.device,min_vocab=config.min_vocab_freq , max_vocab=config.max_vocab_freq)
    input_size , output_size  = len(imdb_loader.src.vocab) , len(imdb_loader.tgt.vocab)
    if original == True:
        model = transformer.encoder_classifier(input_size,output_size,
                                        layers=config.layers,
                                        hidden_size=config.hidden_size,
                                        heads=config.heads,
                                        dropout_p=config.dropout_p,
                                        max_length=config.max_length,
                                        first_norm=config.first_norm)
    elif original == 'github':
        model = transformer_github.encoder_classifier(input_size,output_size,1,1,n_position=config.max_length
                                               ,trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False)


    crit = get_crit(output_size, 0,1)
#    crit = nn.NLLLoss(ignore_index=transformer.pad_token,reduction='sum')
#    crit = nn.CrossEntropyLoss(reduction='mean' )
    optimizer = get_optimizer(model,config)

    if config.device >= 0:
        model.cuda(config.device)
        crit.cuda(config.device)

    trainandvalidate(config,model,crit,optimizer,imdb_loader)

if __name__ == '__main__':
    config = define_arguparser()
    main(config)







