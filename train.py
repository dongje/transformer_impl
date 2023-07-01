import argparse
import torch
import torch.nn as nn
import wmt_dataloader
import transformer
from torch import optim
from torch.cuda.amp import autocast , GradScaler
from tqdm import tqdm
import numpy as np
import nltk.translate.bleu_score as bleu
import torch_optimizer as custom_optim
from torch.optim import Optimizer
import transformer_copy
import transformer_github
import gc


## true , 'copy' , 'torch' , 'github
original = True
#original = 'github'

def define_arguparser():

    p = argparse.ArgumentParser()
    is_continue = False

    p.add_argument('--load_name' , required=is_continue)
    p.add_argument('--init_epoch' , required=is_continue,type=int , default=1)

    p.add_argument('--model_name' , required=not is_continue)
    p.add_argument('--device',required=True,type=int )
    p.add_argument('--train_data', required=True)
    p.add_argument('--valid_data', required=True)
    p.add_argument('--batch_size',type=int , default = 16)
    p.add_argument('--iteration_per_update',type=int,default=16)
    p.add_argument('--autocast',type=int , default=1)
    p.add_argument('--first_norm',type=int,default=1)
    p.add_argument('--n_epochs', type=int,default=15)
    p.add_argument('--max_length',type=int , default=256)
    p.add_argument('--layers', type=int , default = 6)
    p.add_argument('--hidden_size' ,type=int , default = 768)
    p.add_argument('--heads' , type=int,default=8)
    p.add_argument('--dropout_p' ,type=float , default=0.1)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--only_encoder', type=int, default=0)
    p.add_argument(
        '--varnilar_transformer',
        type=int,
        default=1
    )

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

#get criterion function
def get_crit(output_size, pad_index,vocab):

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

#get optimizer
def get_optimizer(model, config):
    if config.use_adam:
        if config.first_norm:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98),eps=1e-9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    return optimizer

def save_model(blue_score,loss,epoch, model,optimizer, config, src_vocab, tgt_vocab):

    # Set a filename for model of last epoch.
    # We need to put every information to filename, as much as possible.
    model_fn = config.model_name

    model_fn = [model_fn[:-1]] + ['%02d.' % epoch,
                                '%.3f' % blue_score , '%.3f'%loss,
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

@torch.no_grad()
def generate_decoder_future_mask(x,input_y): # |bs , n |
    with torch.no_grad():
        mask = torch.triu(x.new_ones((input_y.size(1),input_y.size(1))),diagonal=1).bool()
        return mask.unsqueeze(0).expand(*input_y.size() , input_y.size(1))  ##size : |batch_size , n , n |

@torch.no_grad()
def generate_decoder_future_mask_nn(x,input_y): # |bs , n |
    with torch.no_grad():
        return torch.triu(x.new_ones((input_y.size(1),input_y.size(1))),diagonal=1).bool()

def generate_pad_mask(query_x , keyandvalue_y): # |bs , m|  ,  |bs , n|
    with torch.no_grad():
        mask_key = (keyandvalue_y == transformer.pad_token)  ##keyandvalue는 pad_token이면 true로 나타남
        mask_ = mask_key.unsqueeze(1).expand(*query_x.size(),keyandvalue_y.size(1))
        return mask_[0]   ## size : |m , n|

def patch_trg(trg, pad_idx):
#    trg = trg.transpose(0, 1)
    trg, gold = trg[:,:-1], trg[:,1:]
    return trg, gold


def trainandvalidate(config,model,crit,optimizer,dataloader,scheduler = None):

    device =  next(model.parameters()).device
    if config.device >=0 and bool(config.autocast):
        scaler = GradScaler()

    for epoch in range(config.init_epoch,config.n_epochs):

        model.train()
        for i,mini_batch in tqdm(enumerate(dataloader.train_iter)):
            if i % config.iteration_per_update == 0:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
            if max(mini_batch.src.size(1),mini_batch.trg.size(1)) > config.max_length :
                print(f'{max(mini_batch.src.size(1),mini_batch.trg.size(1))} max length')
                continue

            mini_batch.src = mini_batch.src.to(device)
            mini_batch.trg = mini_batch.trg.to(device)

            x,y = mini_batch.src , mini_batch.trg
            trg , gold = patch_trg(y,transformer.pad_token)
            with autocast(bool(config.autocast)):

                # print(f'trg {trg.size()}')
                # print(trg)
                # print(f'gold {gold.size()}')
                # print(gold)
                # print(f'y {y.size()}')
                # print(y)
                if original == True:
                    y_hat = model(x,trg)   # y_hat = |batch_size , m , output_size| (softmaxed)
                elif original =='copy':
                    y_hat = model.forward(x,mini_batch.trg[:,:-1],None,generate_decoder_future_mask(x,mini_batch.trg[:,:-1]))
                elif original == 'torch':
                    y_hat = model(x.transpose(0,1),mini_batch.trg[:,:-1].transpose(0,1),src_mask = generate_pad_mask(x,x) , tgt_mask = generate_decoder_future_mask_nn(mini_batch.trg[:,:-1],mini_batch.tgt[:,:-1]))
                elif original == 'github':
                    y_hat = model(x,trg)

                # print(f'argmax {torch.argmax(y_hat,dim=-1).size()}')
                print(torch.argmax(y_hat,dim=-1)[:,:5])
                print(gold[:,:5])
                # print(f'y_hat {y_hat.size()}')
                # print(y_hat)

                loss = crit(
                    y_hat.contiguous().view(-1 , y_hat.size(-1)),    # | bs * m , output_size|
                    gold.contiguous().view(-1)                 # |bs * m|
                )
                backward_target = loss.div(y_hat.size(0)).div(config.iteration_per_update)
#                backward_target=loss

                torch.cuda.empty_cache()
                gc.collect()
                mini_batch.src.to('cpu')
                mini_batch.trg.to('cpu')

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

            loss = loss / word_count(y)
            ppl = np.exp(loss.detach().cpu().numpy())
            print(f'train epoch : {epoch} iter : {i} loss : {loss} ppl : {ppl}')

        #valid
        with torch.no_grad():
            model.eval()
            bleu_score = 0
            total_loss = 0
            for i, mini_batch in tqdm(enumerate(dataloader.valid_iter)):
                if max(mini_batch.src.size(1), mini_batch.trg.size(1)) > config.max_length:
                    print(f'{max(mini_batch.src.size(1), mini_batch.trg.size(1))} max length')
                    continue
                mini_batch.src = mini_batch.src.to(device)
                mini_batch.trg = mini_batch.trg.to(device)

                x, y = mini_batch.src, mini_batch.trg[:, 1:]

                with autocast(bool(config.autocast)):
                    y_hat = model(x,mini_batch.trg[:,:-1])
                    loss = crit(
                        y_hat.contiguous().view(-1 , y_hat.size(-1)),
                        y.contiguous().view(-1)
                    )

                y_hat_result = torch.argmax(y_hat,dim=-1)
                print(i)
                tgt_sen = decode_tgt(y,dataloader.trg)
                yhat_sen = decode_tgt(y_hat_result,dataloader.trg)
                score = 0
                for index,b in enumerate(y_hat_result):
                    # print(index)
                    # print(y)
                    # print(y_hat_result)
                    score += bleu.sentence_bleu(yhat_sen[index],tgt_sen[index])
                bleu_score += score / len(y_hat_result)

                loss = loss / word_count(y)
                total_loss += loss.item()
                ppl = np.exp(loss.detach().cpu().numpy())
                print(f'epoch{epoch } valid iter : {i} loss : {loss} ppl : {ppl} bleu : {score / len(y_hat_result)}')

            # print(f'bleu : {bleu_score}')
            if scheduler is not None:
                scheduler.step()
            print(f'{epoch}epoch bleu : {bleu_score}')
        save_model(bleu_score,total_loss,epoch,model,optimizer,config,
                   src_vocab=dataloader.src.vocab,
                   tgt_vocab=dataloader.trg.vocab)


def word_count(tensor):
    return tensor[tensor != transformer.pad_token].bool().sum()

#decode tensor to target word
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

#learning rate scheduler
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

#get model size
def cal_model_size(model):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    return mem_params + mem_buffers


def main(config):

    print(config)
    config.first_norm = bool(config.first_norm)

    #wmt data load
    wmt_loader = wmt_dataloader.wmtloader(config.train_data,config.valid_data,
                       max_length=config.max_length,
                        batch_size=config.batch_size,
                        device = -1)
    # wmt_loader = data.koenloader('dataset/corpus.shuf.train_test.tsv', 'dataset/corpus.shuf.valid_test.tsv',
    #                              max_length=config.max_length,
    #                     batch_size=config.batch_size,
    #                     device=config.device)
    input_size , output_size = len(wmt_loader.src.vocab) , len(wmt_loader.trg.vocab)
    print(f'input size : {input_size} , output size : {output_size}')
    if original == True:
        if bool(config.only_encoder) : #for classify
            model = transformer.encoder_classifier(input_size,output_size,
                                                   layers=config.layers,
                                                   hidden_size=config.hidden_size,
                                                   heads=config.heads,
                                                   dropout_p=config.dropout_p,
                                                   max_length=config.max_length,
                                                   first_norm=config.first_norm
                                                   )
        else:   #for decoder
            model = transformer.Transformer(input_size,output_size,
                                            layers=config.layers,
                                            hidden_size=config.hidden_size,
                                            heads=config.heads,
                                            dropout_p=config.dropout_p,
                                            max_length=config.max_length,
                                            first_norm=config.first_norm,
                                            varnilar = bool(config.varnilar_transformer))
    elif original == 'copy':    #get transformer from a blog
        model = transformer_copy.make_model(input_size,output_size,config.layers,config.hidden_size,4*config.hidden_size)
    # elif original =='torch':
    #     model = nn.Transformer(d_model=config.hidden_size ,nhead=config.heads, num_encoder_layers=config.layers ,num_decoder_layers=config.layers
    #                            ,dim_feedforward=4 * config.hidden_size)
    elif original =='github':   #get transformer official transformer github
        model = transformer_github.Transformer(input_size,output_size,transformer.pad_token,transformer.pad_token,
                                               d_word_vec=config.hidden_size, d_model=config.hidden_size, d_inner=config.hidden_size * 4,
                                               n_layers=config.layers, n_head=config.heads, d_k=config.hidden_size//config.heads,
                                               d_v=config.hidden_size//config.heads, dropout=0.1, n_position=config.max_length,
                                               trg_emb_prj_weight_sharing=False, emb_src_trg_weight_sharing=False)


#    crit = get_crit(output_size, transformer.pad_token)
    crit = nn.NLLLoss(ignore_index=transformer.pad_token,reduction='sum')
#    crit = nn.CrossEntropyLoss(ignore_index=transformer.pad_token , reduction='mean' )
    optimizer = get_optimizer(model,config)
    lr_scheduler = get_scheduler(optimizer, config)
    print('model size : %.3f mb' % (cal_model_size(model)/1024/1024))

    if config.device >= 0:
        model.cuda(config.device)
        crit.cuda(config.device)

    trainandvalidate(config,model,crit,optimizer,wmt_loader,lr_scheduler)

if __name__ == '__main__':
    config = define_arguparser()
    main(config)






