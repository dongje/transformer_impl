
from torch import nn
import torch
import wmt_dataloader
import torchtext
import random
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data

import nltk.translate.bleu_score as bleu
from torchtext.data.utils import get_tokenizer

eos_token = 3
pad_token = 1
bos_token = 2

## encoder,decoder에서 multi attention을 수행하기 위한 class
class MultiheadAttention(nn.Module):

    def __init__(self,
                 heads,
                 hidden_size):
        super().__init__()
        self.dk = hidden_size // heads
        self.hidden_size = hidden_size
        self.heads = heads

        self.W_Q = nn.Linear(hidden_size,hidden_size)
        self.W_K = nn.Linear(hidden_size,hidden_size)
        self.W_V = nn.Linear(hidden_size,hidden_size)
        self.W_last = nn.Linear(hidden_size,hidden_size)
        self.softmax=nn.Softmax(dim=-1)


    def forward(self,Query_x , KeyandValue_x,mask = None,train_decode_masking = False):

        #head 수 만큼으로 분할
        Q = self.W_Q(Query_x).split(self.hidden_size//self.heads,dim = -1)   # Q heads * |bs , m , dk|
        K = self.W_K(KeyandValue_x).split(self.hidden_size//self.heads , dim=-1)  # K heads * |bs , n , dk|
        V = self.W_V(KeyandValue_x).split(self.hidden_size//self.heads, dim=-1)  # V heads * |bs , n , dk|

        #한 번에 연산하기 위해
        Q = torch.cat(Q , dim = 0)      # Q | heads * bs , m , dk|
        K = torch.cat(K , dim = 0)      # K | heads * bs , n , dk|
        V = torch.cat(V , dim = 0)      # V | heads * bs , n , dk|

        QKmatmuled = torch.bmm(Q, torch.transpose(K, 1, 2))   # | heads * bs , m , n |
        #주로 인코더에서 쓰이는 입력x에대한 패딩 마스크 (소프트맥스를 0으로 만들기 위한)
        #또는 decoder 학습모드에서 다음 timestep을 보는걸 방지하기 위한
        if mask is not None:   #mask = |batch_size , m , n|
            mask = torch.cat([mask for _ in range(self.heads)] , dim = 0)    #mask = | batch_size * heads , m , n |
            assert QKmatmuled.size() == mask.size()
            QKmatmuled.masked_fill_(mask,-float('inf'))

        k = 1
        softmaxQK = self.softmax(QKmatmuled/(self.dk ** .5)/k)   # | heads * bs , m , n |
        ##attention변형
        softmaxQK = self.moderate_attention(softmaxQK)

        result_v = torch.bmm(softmaxQK,V)        # | heads * bs , m , dk |
        result = torch.cat(result_v.split( Query_x.size(0) , dim = 0),dim = -1)  # | bs , m , dk * heads |

        return self.W_last(result) # | bs , m , hidden_size |

    def moderate_attention(self,softmaxQK):

        #token_length = softmaxQK.size()[1]
        mode = 'stocastic'

        if mode == 'original':
            return softmaxQK
        elif mode == 'cumulative':
            cumul_p = 0.9

            sorted_a, sorted_index = torch.sort(softmaxQK, dim=-1)
            cumsum = torch.cumsum(sorted_a, dim=-1)
            cum_tensor = (cumsum >= 1 - cumul_p)
            true_position = torch.nonzero(cum_tensor, as_tuple=False)
            false_position = torch.nonzero(~cum_tensor, as_tuple=False)
            sorted_index[true_position[:, 0], true_position[:, 1] , true_position[:,2]] = -1

            changed_sorted_index = sorted_index.clone()
            for i in range(0, sorted_index.size(0)):
                for j in range(0,sorted_index.size(1)):
                    changed_sorted_index[i][j][sorted_index[i][j] == -1] = sorted_index[i][j][0]

            changed_softmaxQK = softmaxQK.clone()
            for i in range(0, softmaxQK.size()[0]):
                for j in range(0,softmaxQK.size()[1]):
                    changed_softmaxQK[i][j][changed_sorted_index[i][j]] = float('-inf')
            changed_softmaxQK = torch.softmax(changed_softmaxQK,dim=-1)

            return changed_softmaxQK

        elif mode == 'threshold':
            threshold = 0.001
            softmaxQK[softmaxQK<threshold] = float('-inf')
            softmaxQK = torch.softmax(softmaxQK , dim=-1)

            return softmaxQK

        elif mode =='top-k':
            scale = 1.3
            k = 5
            values , indices = torch.topk(softmaxQK , k=k,dim=-1)
            softmaxQK.scatter_(-1, indices, torch.mul(softmaxQK,scale).gather(-1, indices))
            scaled_sum = torch.sum(softmaxQK,dim=-1)
            softmaxQK /= scaled_sum.unsqueeze(-1)
            return softmaxQK

        elif mode =='top-p':
            length = softmaxQK.size()[-1]
            p = 0.05
            scale = 1.2
            k = round(p * length)
            values, indices = torch.topk(softmaxQK, k=k, dim=-1)
            softmaxQK.scatter_(-1, indices, torch.mul(softmaxQK, scale).gather(-1, indices))
            scaled_sum = torch.sum(softmaxQK, dim=-1)
            softmaxQK /= scaled_sum.unsqueeze(-1)
            return softmaxQK

        elif mode =='stocastic':
            softmax_size = softmaxQK.size()
            length = softmax_size[-1]
            p = 0.05
            scale = 1.2
            k = round(p * length)
            randidx = []
            rand_p = 0.1
            for i in range(round(rand_p * length)):
                randidx.append(random.randint(0,length-1))
            values, indices = torch.topk(softmaxQK, k=k, dim=-1)
            broaden = torch.tensor(randidx).to(torch.device("cuda")).repeat(softmax_size[0] * softmax_size[1] , 1)
            broaden = broaden.view(softmax_size[0] , softmax_size[1] , len(randidx))
            indices  = torch.cat([indices , broaden] , dim=-1)
            softmaxQK.scatter_(-1, indices, torch.mul(softmaxQK, scale).gather(-1, indices))
            scaled_sum = torch.sum(softmaxQK, dim=-1)
            softmaxQK /= scaled_sum.unsqueeze(-1)
            return softmaxQK







#1개 단위의 인코더 블록 클래스
class EncoderBlock(nn.Module):

    def __init__(
            self,
            hidden_size,
            heads,
            dropout_p = 0.1,
            leaky_relu = False,
            first_norm = True
        ):
        super().__init__()
        self.first_norm = first_norm
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.multihead_att = MultiheadAttention(heads,hidden_size)

    def forward(self,x_s,pad_mask):    # pad부분 소프트맥스를 -inf로 채우기 위한 마스크 첨가
        '''size = (batch_size , n , hidden_size)'''

        ##only decoder p transformer
#        x = x_s[-1]
        x = x_s
        #정규화 순서에따라 다른 순서로 계산
        if self.first_norm:
            final1 = self.layer_norm(x)
            final1 = x + self.dropout(self.multihead_att(final1,final1,pad_mask))

            final2 = self.FFN(self.layer_norm(final1))
            final2 = final1 + self.dropout(final2)

        else:
            final1 = self.multihead_att(x,x,pad_mask)
            final1 = self.dropout(self.layer_norm(final1 + x))

            final2 = self.FFN(final1)
            final2 = self.dropout(self.layer_norm(final2 + final1))

#       only decoder p-transformer
#        x_s.append(final2)
#        return x_s , pad_mask   # encoder return size . |bs , n , dk * heads|
        return final2 , pad_mask

#decoder 한 개 단위의 블록 클래스
class DecoderBlock(nn.Module):

    def __init__(
            self,
            hidden_size,
            heads,
            dropout_p=0.1,
            leaky_relu=False,
            first_norm=True,
            vinalar = True
        ):
        super().__init__()
        self.first_norm = first_norm
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.multihead_att1 = MultiheadAttention(heads, hidden_size)
        self.multihead_att2 = MultiheadAttention(heads, hidden_size)

    def forward(self,x,encoder_x ,pad_mask,future_mask,prev):
        # encoder_x = |bs , m , hidden| from encoder
        # x = |bs , n , hidden| or |bs , n , 1|
        # prev = |bs,t-1,hidden_size| or None
        # pad_mask = |bs , n , m|   디코더 input에서 인코더의 pad값에대한 마스크
        # future_mask = |bs , n , n| or None

        #정규화를 먼저 할 경우
        if self.first_norm:
            if prev is None:  ## 학습모드
                # x = |bs, n , hidden_size|
                #final1 = |bs , n , hidden_size|
                final1 = self.layer_norm(x)
                final1 = x + \
                         self.dropout(self.multihead_att1(final1, final1, mask=future_mask))  # decoder에서 미래의 답 보는거 방지
            else:  ## search 모드  t번째.
                # x = |bs , 1 , hidden_size|
                # prev = |bs , t-1 , hidden_size|
                # final1 = |bs , 1 , hidden_size|
                normed_prev = self.layer_norm(prev)
                final1 = self.layer_norm(x)
                final1 = x +\
                          self.dropout(self.multihead_att1(final1, normed_prev, mask=None))

            final2 = self.layer_norm(final1)
            final2 = final1 + self.dropout(self.multihead_att2(final2, encoder_x, pad_mask))

            final3 = self.layer_norm(final2)
            final3 = final2 + self.dropout(self.FFN(final3))

            return final3, encoder_x, pad_mask, future_mask, prev

        #정규화 나중에 할 경우
        elif self.first_norm == False:
            if prev is None:  ## 학습모드
                final1 = self.dropout(self.multihead_att1(x,x,mask = future_mask)) #decoder에서 미래의 답 보는거 방지
                final1 = self.dropout(self.layer_norm(final1 + x))
            else :    ## search 모드
                final1 = self.dropout(self.multihead_att1(x, prev, mask = None))
                final1 = self.dropout(self.layer_norm(final1 + x))

            final2 = self.dropout(self.multihead_att2(final1,encoder_x,pad_mask))
            final2 = self.dropout(self.layer_norm(final2 + final1))

            final3 = self.dropout(self.layer_norm(self.FFN(final2) + final2))

            return final3 , encoder_x , pad_mask , future_mask , prev


# encoder , decoder에대해 튜플 입력을 받기 위한 nn.Sequential 상속 클래스
class encoder_sequential(nn.Sequential):
    def forward(self, *x):
        for module in self._modules.values():      ## block수 만큼 return을 입력으로 그대로 넣어준다
            x = module(*x)
        return x


class decoder_sequential(nn.Sequential):
    def forward(self, *x):
        varnilar = x[-1]
        x = tuple(list(x)[:-1])     #varnilar t/f는 제거
        if varnilar:
            x = list(x)
            x[1] = x[1][-1]
            x = tuple(x)
            for module in self._modules.values():      ## block수 만큼 return을 입력으로 그대로 넣어준다
                x = module(*x)
            return x
        else:
            encoders = x[1]
            for i , module in enumerate(self._modules.values()):  #encoder , decode의 길이는 같아야함
                x = list(x)
                x[1] = encoders[i+1]  #처음거는 encoder의 처음 inputs
                x = tuple(x)
                x = module(*x)
            return x


class Transformer(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 layers=6,
                 hidden_size=768,
                 heads=8,
                 dropout_p = 0.1,
                 max_length = 256,
                 first_norm = True,
                 varnilar = True,
                 use_leaky_relu = False,
                 ):

        super().__init__()
        self.varnilar = varnilar
        self.hidden_size = hidden_size
        self.heads = heads
        self.dk = hidden_size // heads
        self.max_length = max_length

        self.encoder_emb = nn.Embedding(input_size,hidden_size)
        self.decoder_emb = nn.Embedding(output_size,hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        encoder_layers = [EncoderBlock(hidden_size,heads,dropout_p=0.1,leaky_relu=use_leaky_relu,first_norm=first_norm)\
                          for i in range(layers)]
        self.encoders = encoder_sequential(*encoder_layers)

        decoder_layers = [DecoderBlock(hidden_size, heads, dropout_p=0.1, leaky_relu=use_leaky_relu,first_norm=first_norm) \
                          for i in range(layers)]
        self.decoders = decoder_sequential(*decoder_layers)

        if first_norm:
            self.generator = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.generator = nn.Sequential(
                nn.Linear(hidden_size,output_size),
                nn.LogSoftmax(dim = -1)
            )

        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

    # x,y에대한 임베딩 벡터에 대해 순서에대한 정보를 나타내는 encoding 생성
    @torch.no_grad()
    def generate_pos_encoding(self,bs):
        pos_encoding = torch.zeros(self.max_length,self.hidden_size,dtype=torch.float)   # |length , hidden_size|

        pos = torch.arange(self.max_length).unsqueeze(1).float()  ## |max_length , 1|
        i = (torch.arange(self.hidden_size//2).unsqueeze(0).float()) * 2   ## |1 , hidden_size//2 |

        pos_encoding[:,0::2] = torch.sin(pos / 10000**(i/self.hidden_size))
        pos_encoding[:,1::2] = torch.cos(pos / 10000**(i/self.hidden_size))

        return pos_encoding.unsqueeze(0).expand(bs,self.max_length,self.hidden_size)   # |batch_size , full_length , hidden_size|

    def get_pos_encoding(self,bs,pos_enc,len,start_pos=0):
        assert len + start_pos <= self.max_length

        return pos_enc[:,start_pos:len+start_pos,:]

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size, max_length):
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))   #이게 어떻게 가능함??
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc

    def _position_encoding(self, x, init_pos=0):
        # |x| = (batch_size, n, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)   ## 이게 어떻게 가능함?
        # |pos_enc| = (1, n, hidden_size)
        x = x + pos_enc.to(x.device)

        return x

    # 인코더,디코더에서 멀티헤드어텐션시 input에대한 pad을 -inf로 채우기위한 마스크
    @torch.no_grad()
    def generate_pad_mask(self,query_x , keyandvalue_y): # |bs , m|  ,  |bs , n|
        with torch.no_grad():
            mask_key = (keyandvalue_y == pad_token)  ##keyandvalue는 pad_token이면 true로 나타남  |bs , n|
            mask_ = mask_key.unsqueeze(1).expand(*query_x.size(),keyandvalue_y.size(1))
            return mask_   ## size : |batch_size , m , n|

    # 디코더에서 학습 시 n번째 timestep에서 n+1번째 timestep과 어텐션 하는걸 방지하는 마스크 생성
    @torch.no_grad()
    def generate_decoder_future_mask(self,x,input_y): # |bs , n |
        with torch.no_grad():
            mask = torch.triu(x.new_ones((input_y.size(1),input_y.size(1))),diagonal=1).bool()
            return mask.unsqueeze(0).expand(*input_y.size() , input_y.size(1))  ##size : |batch_size , n , n |


    def forward(self,x,y): # x |batch_size , m |  y |batch_size , n |   type of indice

        with torch.no_grad():
            pos_enc = self.generate_pos_encoding(x.size(0))

            encoder_mask = self.generate_pad_mask(x,x)
            decoder_mask = self.generate_pad_mask(y,x)
        x = self.encoder_emb(x)   # x = | batch_size , m , hidden_size|
        x = self.dropout( x + self.get_pos_encoding(x.size(0),pos_enc,x.size(1)).to(x.device) ) # x = |bs , m , hidden_size|
        x_s = [x]
        encoder_results,_ = self.encoders(x_s,encoder_mask)   # encoder_result =  ( layer+1 ) * |bs , m , hidden_size|
        with torch.no_grad():
            future_mask = self.generate_decoder_future_mask(x,y)
        y = self.decoder_emb(y)   # y = |batch_size , n , hidden_size|
        y = y + self.get_pos_encoding(y.size(0),pos_enc,y.size(1)).to(x.device)   # y = |bs , n , hidden_size|
        decoder_result,_1,_2,_3,_4 = self.decoders( self.dropout(y) , encoder_results,decoder_mask, future_mask,None,self.varnilar)
        y_hat = self.generator(decoder_result)

        return y_hat


    def search(self,x,bos_token_id = 1):   # x  | batch_size , m |

        pos_enc = self.generate_pos_encoding(x.size(0))
        y = (torch.zeros(x.size(0),1) + bos_token).long().to(x.device)
        encoder_mask = self.generate_pad_mask(x,x)
        decoder_pad_mask = self.generate_pad_mask(y, x)   # (bs , 1 , m)

        x = self.encoder_emb(x)   # | batch_size , m , hidden|
        x = x + self.get_pos_encoding(x.size(0),pos_enc,x.size(1)).to(x.device)
        encoder_results,_ = self.encoders([x], encoder_mask)  # encoder_result = |bs , m , hidden_size|

        indices = torch.zeros(x.size(0),self.max_length).fill_(pad_token).long()

        decoding_sum = torch.zeros(x.size(0),1).fill_(False).bool().to(x.device)
        prevs = [None for _ in range(len(self.decoders.__module__)+1)]

        y_hats = None
        t = 0
        import gc
        while decoding_sum.sum() < x.size(0) and t < self.max_length:
            y = self.decoder_emb(y)
            y = self.dropout(
                y + self.get_pos_encoding(y.size(0),pos_enc,1,t).to(x.device)
            ) # y = (bs , 1 , hidden_size)

            if prevs[0] is None:
                prevs[0] = y
            else:
                prevs[0] = torch.cat([prevs[0] , y],dim = 1)

            for layer_i , decoder_layer_i in enumerate(self.decoders._modules.values()):

                torch.cuda.empty_cache()
                gc.collect()

                prev = prevs[layer_i]   # (bs , t , hidden_size)
                if self.varnilar:
                    y,_,_,_,_ = decoder_layer_i(y,encoder_results[-1],decoder_pad_mask,None,prev)
                    # (bs , 1 , hidden_size)
                else :
                    y, _, _, _, _ = decoder_layer_i(y, encoder_results[layer_i+1], decoder_pad_mask, None, prev)

                if prevs[layer_i+1] is None:
                    prevs[layer_i+1] = y
                else:
                    prevs[layer_i+1] = torch.cat([prevs[layer_i+1],y],dim = 1)
                # (bs , t , hidden_size)

            y_hat_t = self.generator(y)    # (bs , 1 , d_output_size)
            if y_hats is None:
                y_hats = y_hat_t
            else:
                y_hats = torch.cat([y_hats,y_hat_t],dim=1)
            indice = torch.argmax(y_hat_t,dim = -1) # (bs , 1) indice

            indice = indice.masked_fill_(decoding_sum , pad_token)
            indices[:,t] = indice.view(-1)

            decoding_sum += (indice == eos_token)

            t += 1
            y = indice

        return indices ,y_hats

#for only classify task
class encoder_classifier(nn.Module):

    def __init__(self,
                 input_size,output_size,
                 layers=6,
                 hidden_size=768,
                 heads=8,
                 dropout_p = 0.1,
                 max_length = 256,
                 first_norm = False,
                 use_leaky_relu = False,
                 ):

        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.dk = hidden_size // heads
        self.max_length = max_length

        self.encoder_emb = nn.Embedding(input_size,hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        encoder_layers = [EncoderBlock(hidden_size,heads,dropout_p=0.1,leaky_relu=use_leaky_relu)\
                          for i in range(layers)]
        self.encoders = encoder_sequential(*encoder_layers)

        if first_norm:
            self.generator = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.generator = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.LogSoftmax(dim = -1)
            )


        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

    # x,y에대한 임베딩 벡터에 대해 순서에대한 정보를 나타내는 encoding 생성
    @torch.no_grad()
    def generate_pos_encoding(self,bs):
        pos_encoding = torch.zeros(self.max_length,self.hidden_size,dtype=torch.float)   # |length , hidden_size|

        pos = torch.arange(self.max_length).unsqueeze(1).float()  ## |max_length , 1|
        i = (torch.arange(self.hidden_size//2).unsqueeze(0).float()) * 2   ## |1 , hidden_size//2 |

        pos_encoding[:,0::2] = torch.sin(pos / 10000**(i/self.hidden_size))
        pos_encoding[:,1::2] = torch.cos(pos / 10000**(i/self.hidden_size))

        return pos_encoding.unsqueeze(0).expand(bs,self.max_length,self.hidden_size)   # |batch_size , full_length , hidden_size|

    def get_pos_encoding(self,bs,pos_enc,len,start_pos=0):
        assert len + start_pos <= self.max_length

        return pos_enc[:,start_pos:len+start_pos,:]

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size, max_length):
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))   #이게 어떻게 가능함??
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc

    def _position_encoding(self, x, init_pos=0):
        # |x| = (batch_size, n, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)   ## 이게 어떻게 가능함?
        # |pos_enc| = (1, n, hidden_size)
        x = x + pos_enc.to(x.device)

        return x

    # 인코더,디코더에서 멀티헤드어텐션시 input에대한 pad을 -inf로 채우기위한 마스크
    @torch.no_grad()
    def generate_pad_mask(self,query_x , keyandvalue_y): # |bs , m|  ,  |bs , n|
        with torch.no_grad():
            mask_key = (keyandvalue_y == pad_token)  ##keyandvalue는 pad_token이면 true로 나타남
            mask_ = mask_key.unsqueeze(1).expand(*query_x.size(),keyandvalue_y.size(1))
            return mask_   ## size : |batch_size , m , n|

    def forward(self,x):
        with torch.no_grad():
            pos_enc = self.generate_pos_encoding(x.size(0))
            encoder_mask = self.generate_pad_mask(x, x)
        x = self.encoder_emb(x)  # x = | batch_size , m , hidden_size|
        x = x + self.get_pos_encoding(x.size(0), pos_enc, x.size(1)).to(x.device)  # x = |bs , m , hidden_size|
        encoder_result, _ = self.encoders(self.dropout(x), encoder_mask)  # encoder_result = |bs , m , hidden_size|
        y_hat = self.generator(encoder_result)   # y_hat = |bs , m , output_size|


        return y_hat[:,0,:]    ##  y_hat = |bs , input_size|  bos token만 return




##############################
##############################
##############################
##############################  JUST FOR TEST
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################









#
#
# def decode_tgt(y_tensor,tgt_field): ### bs , n
#
#     index_to_word_list = tgt_field.vocab.itos
#     splited = []
#     for sen in y_tensor:
#         sentence = []
#         sen = sen[sen != pad_token]
#         for idx in sen:
#             sentence.append(index_to_word_list[idx])
#         splited.append(sentence)
#
#     return splited
#
# def test():
#     wmt_loader = wmt_dataloader.wmtloader('dataset/wmt16_train_10m','dataset/wmt16_valid',
#                            max_length=64,
#                             batch_size=16,
#                             device=-1)
#     input_size , output_size = len(wmt_loader.src.vocab) , len(wmt_loader.trg.vocab)
#     model = Transformer(input_size,output_size,
#                                     layers=6,
#                                     hidden_size=768,
#                                     heads=8,
#                                     dropout_p=0.1,
#                                     max_length=256 ,
#                                     first_norm=True)
#
#     for i , mini_batch in enumerate(wmt_loader.train_iter):
#         mini_batch.src = mini_batch.src.to('cpu')
#         mini_batch.trg = mini_batch.trg.to('cpu')
#
#         x, y = mini_batch.src, mini_batch.trg[:, 1:]
#
#         y_hat = model(x,mini_batch.trg[:, :-1])
#         y_result = torch.argmax(y_hat,dim=-1)
#         print(i)
#         print(y)
#         print(y_result)
#
#         pass
#
# def define_field():
#
#
#     en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#     de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
#     '''
#     To avoid use DataLoader class, just declare dummy fields.
#     With those fields, we can retore mapping table between words and indice.
#     '''
#     return (
#          data.Field(
#              sequential=True,
#              use_vocab=True,
#              batch_first=True,
#              #            include_lengths=True,
#              lower=True,
#              #            init_token='<BOS>',
#              #            eos_token='<EOS>',
#              tokenize=de_tokenizer
#         ),
#          data.Field(
#              sequential=True,
#              use_vocab=True,
#              batch_first=True,
#              #            include_lengths=True,
#              #        init_token='<BOS>',
#              eos_token='<EOS>',
#              tokenize=en_tokenizer
#          )
# )
#
#
# def test_load( train_config, src_field, trg_field):
#     test_data = wmt_dataloader.TranslationDataset_separated(
#         path='dataset/wmt16_test',
#         fields=[('src', src_field), ('trg', trg_field)],
#         max_length=train_config.max_length
#     )
#     test_iter = data.BucketIterator(
#         test_data,
#         batch_size=16,
#         device='cpu',
#         shuffle=True,
#         sort_key=lambda x: len(x.src) + (train_config.max_length * len(x.trg)),
#         sort_within_batch=True,
#     )
#     return test_iter
#
# def testtest():
#
#     saved_data = torch.load(
#         'realmin.04..0.000.829.274.e',
#         map_location='cpu'
#     )
#
#     train_config = saved_data['config']
#     saved_model = saved_data['model']
#     src_vocab = saved_data['src_vocab']
#     tgt_vocab = saved_data['tgt_vocab']
#
#     model = Transformer(len(src_vocab), len(tgt_vocab),
#                                     layers=train_config.layers,
#                                     hidden_size=train_config.hidden_size,
#                                     heads=train_config.heads,
#                                     dropout_p=train_config.dropout_p,
#                                     max_length=train_config.max_length,
#                                     first_norm=train_config.first_norm)
#
#     model.load_state_dict(saved_model)
#
#
#
#     src_field, tgt_field = define_field()
#     src_field.vocab = src_vocab
#     tgt_field.vocab = tgt_vocab
#     test_loader = test_load(train_config,src_field,tgt_field)
#
#     crit = nn.NLLLoss(ignore_index=1,reduction='sum')
#
#     def patch_trg(trg, pad_idx):
#         #    trg = trg.transpose(0, 1)
#         trg, gold = trg[:, :-1], trg[:, 1:]
#         return trg, gold
#     bleu_score = 0
#     for i, mini_batch in enumerate(test_loader):
#         trg, gold = patch_trg(mini_batch.trg, pad_token)
#
#         indices , y_hats = model.search(mini_batch.src)
#         for g,y in zip(gold,indices):
#             print(f'gold {g}')
#             print(f'y {y}')
#
#         tgt_sen = decode_tgt(gold, tgt_field)
#         yhat_sen = decode_tgt(indices, tgt_field)
#         score = 0
#         for index, b in enumerate(yhat_sen):
#             print(' '.join(tgt_sen[index]))
#             print(' '.join(yhat_sen[index]))
#             score += bleu.sentence_bleu(yhat_sen[index], tgt_sen[index])
#         bleu_score += score / len(yhat_sen)
#         print(score / len(yhat_sen))



#testtest()
#test()


