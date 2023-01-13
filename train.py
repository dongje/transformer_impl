import argparse
import torch
import torch.nn as nn
import data
import transformer
from torch import optim
from torch.cuda.amp import autocast , GradScaler
from tqdm import tqdm
import numpy as np
import nltk.translate.bleu_score as bleu
import torch_optimizer as custom_optim

def define_arguparser():

    p = argparse.ArgumentParser()
    is_continue = False

    p.add_argument('--load_name' , required=is_continue)
    p.add_argument('--init_epoch' , required=is_continue,type=int , default=1)

    p.add_argument('--model_name' , required=not is_continue)
    p.add_argument('--device',required=True,type=int )
    p.add_argument('--batch_size',type=int , default = 16)
    p.add_argument('--iteration_per_update',type=int,default=16)
    p.add_argument('--autocast',type=bool , default=True)
    p.add_argument('--first_norm',type=bool,default=True)
    p.add_argument('--n_epochs', type=int,default=15)
    p.add_argument('--max_length',type=int , default=256)
    p.add_argument('--layers', type=int , default = 6)
    p.add_argument('--hidden_size' ,type=int , default = 768)
    p.add_argument('--heads' , type=int,default=8)
    p.add_argument('--dropout_p' ,type=float , default=0.1)
    p.add_argument('--lr',type=float,default=1e-3)
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

    config = p.parse_args()

    assert config.hidden_size % config.heads == 0

    return config


def get_crit(output_size, pad_index,vocab):

    itos = []
    for k in vocab.itos:
        itos.append(vocab.freqs[k])
    itos = nn.Softmax(torch.tensor(itos),dim = 0)
    itos[itos == 0] = float('inf')
    itos = 1/itos

    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones_like(output_size)
    loss_weight[pad_index] = 0

    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )
    return crit

def get_optimizer(model, config):
    if config.use_adam:
        if config.first_norm:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    return optimizer


def save_model(blue_score,epoch, model,optimizer, config, src_vocab, tgt_vocab):

    # Set a filename for model of last epoch.
    # We need to put every information to filename, as much as possible.
    model_fn = config.model_name

    model_fn = [model_fn[:-1]] + ['%02d.' % epoch,
                                '%.3f' % blue_score
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

def trainandvalidate(config,model,crit,optimizer,dataloader):

    device =  next(model.parameters()).device
    if config.device >=0:
        scaler = GradScaler()

    for epoch in range(config.init_epoch,config.n_epochs):

        model.train()
        for i,mini_batch in tqdm(enumerate(dataloader.train_iter)):
            if i % config.iteration_per_update == 0:
                optimizer.zero_grad()
            mini_batch.de = mini_batch.de[0].to(device)
            mini_batch.en = mini_batch.en[0].to(device)

            x,y = mini_batch.de , mini_batch.en[:,1:]
            with autocast(config.autocast):
                y_hat = model(x,mini_batch.en[:,:-1])
                print(y)
                print(torch.argmax(y_hat,dim=-1))
                print(y_hat)

                loss = crit(
                    y_hat.contiguous().view(-1 , y_hat.size(-1)),
                    y.contiguous().view(-1)
                )
                backward_target = loss.div(y_hat.size(0))

                if config.device >= 0 and config.autocast:
                    scaler.scale(backward_target).backward()
                else:
                    backward_target.backward()

            if i % config.iteration_per_update == 0:
                if config.device >= 0 and config.autocast:
                # Use scaler instead of optimizer.step() if using GPU.
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            loss = loss / word_count(y)
            ppl = np.exp(loss.detach().cpu().numpy())
            print(f'train iter : {i} loss : {loss} ppl : {ppl}')

        #valid
        with torch.no_grad():
            model.eval()
            bleu_score = 0
            for i, mini_batch in tqdm(enumerate(dataloader.valid_iter)):
                mini_batch.de = mini_batch.de[0].to(device)
                mini_batch.en = mini_batch.en[0].to(device)

                x, y = mini_batch.de, mini_batch.en[:, 1:]

                with autocast(config.autocast):
                    y_hat = model(x,mini_batch.en[:,:-1])
                    loss = crit(
                        y_hat.contiguous().view(-1 , y_hat.size(-1)),
                        y.contiguous().view(-1)
                    )

                y_hat_result = torch.argmax(y_hat,dim=-1)
                print(i)
                tgt_sen = decode_tgt(y,dataloader.tgt)
                yhat_sen = decode_tgt(y_hat_result,dataloader.tgt)
                score = 0
                for index,b in enumerate(y_hat_result):
                    # print(index)
                    # print(y)
                    # print(y_hat_result)
                    score += bleu.sentence_bleu(yhat_sen[index],tgt_sen[index])
                bleu_score += score / len(y_hat_result)

                loss = loss / word_count(y)
                ppl = np.exp(loss.detach().cpu().numpy())
                print(f'valid iter : {i} loss : {loss} ppl : {ppl}')


            print(f'bleu : {bleu_score}')

        save_model(bleu_score,epoch,model,optimizer,config,
                   src_vocab=len(dataloader.src.vocab.itos),
                   tgt_vocab=len(dataloader.tgt.vocab.itos))

def word_count(tensor):
    return tensor[tensor != transformer.pad_token].bool().sum()

def decode_tgt(y_tensor,tgt_field): ### bs , n

    index_to_word_list = tgt_field.vocab.itos
    splited = []
    for sen in y_tensor:
        sentence = []
        sen = sen[sen != transformer.pad_token]
        for idx in sen:
            sentence.append(index_to_word_list[idx])
        splited.append(sentence)

    return splited



def main(config):

    print(config)

    wmt_loader = data.wmtloader('wmt16imsi.csv','wmt16_valid.csv',
                       max_length=255,
                        batch_size=config.batch_size,
                        device=config.device)
    input_size , output_size = len(wmt_loader.src.vocab) , len(wmt_loader.tgt.vocab)
    model = transformer.Transformer(input_size,output_size,
                                    layers=config.layers,
                                    hidden_size=config.hidden_size,
                                    heads=config.heads,
                                    dropout_p=config.dropout_p,
                                    max_length=config.max_length,
                                    first_norm=config.first_norm)
#    crit = get_crit(output_size, transformer.pad_token)
    crit = nn.NLLLoss(ignore_index=transformer.pad_token,reduction='sum')
    optimizer = get_optimizer(model,config)

    if config.device >= 0:
        model.cuda(config.device)
        crit.cuda(config.device)

    trainandvalidate(config,model,crit,optimizer,wmt_loader)

if __name__ == '__main__':
    config = define_arguparser()
    main(config)






