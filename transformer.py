
from torch import nn
import torch

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

        QKmatmuled = torch.matmul(Q, torch.transpose(K, 1, 2))   # | heads * bs , m , n |
        #주로 인코더에서 쓰이는 입력x에대한 패딩 마스크 (소프트맥스를 0으로 만들기 위한)
        #또는 decoder 학습모드에서 다음 timestep을 보는걸 방지하기 위한
        if mask is not None:   #mask = |batch_size , m , n|
            mask = torch.cat([mask for _ in range(self.heads)] , dim = 0)    #mask = | batch_size * heads , m , n |
            assert QKmatmuled.size() == mask.size()
            QKmatmuled.masked_fill_(mask,-float('inf'))

        somtmaxQK = self.softmax(QKmatmuled/(self.dk ** .5))   # | heads * bs , m , n |
        result_v = torch.bmm(somtmaxQK,V)        # | heads * bs , m , dk |
        result = torch.cat(result_v.split( Query_x.size(0) , dim = 0),dim = -1)  # | bs , m , dk * heads |

        return self.W_last(result) # | bs , m , hidden_size |


#1개 단위의 인코더 블록 클래스
class EncoderBlock(nn.Module):

    def __init__(
            self,
            hidden_size,
            heads,
            dropout_p = 0.1,
            leaky_relu = False,
            first_norm = False
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

    def forward(self,x,pad_mask):    # pad부분 소프트맥스를 -inf로 채우기 위한 마스크 첨가
        '''size = (batch_size , n , hidden_size)'''

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

        return final2 , pad_mask   # encoder return size . |bs , n , dk * heads|


#decoder 한 개 단위의 블록 클래스
class DecoderBlock(nn.Module):

    def __init__(
            self,
            hidden_size,
            heads,
            dropout_p=0.1,
            leaky_relu=False,
            first_norm=False
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
class coder_sequential(nn.Sequential):
    def forward(self, *x):
        for module in self._modules.values():      ## block수 만큼 return을 입력으로 그대로 넣어준다
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
                 first_norm = False,
                 use_leaky_relu = False,
                 ):

        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.dk = hidden_size // heads
        self.max_length = max_length

        self.encoder_emb = nn.Embedding(input_size,hidden_size)
        self.decoder_emb = nn.Embedding(output_size,hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        encoder_layers = [EncoderBlock(hidden_size,heads,dropout_p=0.1,leaky_relu=use_leaky_relu)\
                          for i in range(layers)]
        self.encoders = coder_sequential(*encoder_layers)

        decoder_layers = [DecoderBlock(hidden_size, heads, dropout_p=0.1, leaky_relu=use_leaky_relu) \
                          for i in range(layers)]
        self.decoders = coder_sequential(*decoder_layers)

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
            mask_key = (keyandvalue_y == pad_token)  ##keyandvalue는 pad_token이면 true로 나타남
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
        x = x + self.get_pos_encoding(x.size(0),pos_enc,x.size(1)).to(x.device)  # x = |bs , m , hidden_size|
        encoder_result,_ = self.encoders(self.dropout(x),encoder_mask)   # encoder_result = |bs , m , hidden_size|
        with torch.no_grad():
            future_mask = self.generate_decoder_future_mask(x,y)
        y = self.decoder_emb(y)   # y = |batch_size , n , hidden_size|
        y = y + self.get_pos_encoding(y.size(0),pos_enc,y.size(1)).to(x.device)   # y = |bs , n , hidden_size|
        decoder_result,_1,_2,_3,_4 = self.decoders( self.dropout(y) , encoder_result,decoder_mask, future_mask,None)
        y_hat = self.generator(decoder_result)

        return y_hat


    def search(self,x,bos_token_id = 1):   # x  | batch_size , m |

        pos_enc = self.generate_pos_encoding(x.size(0))
        encoder_mask = self.generate_pad_mask(x,x)
        x = self.encoder_emb(x)   # | batch_size , m , Vx|
        x = x + self.get_pos_encoding(x.size(0),pos_enc,x.size(1)).to(x.device)
        encoder_result = self.encoders(x, encoder_mask)  # encoder_result = |bs , m , hidden_size|

        y = torch.zeros(x.size(0),1)
        y[-1] = bos_token_id
        indices = torch.zeros(x.size(0),self.max_length).fill_(pad_token)
        decoder_pad_mask = self.generate_pad_mask(y, x)   # (bs , 1 , m)

        decoding_sum = torch.zeros(x.size(0),1).fill_(False).bool()
        prevs = [None for _ in range(len(self.decoders.__module__)+1)]

        t = 0
        while decoding_sum.sum() < x.size(0) and t < self.max_length:

            y = self.decoder_emb(y)
            y = self.dropout(
                y + self.get_pos_encoding(y.size(0),pos_enc,1,t).to(x.device)
            ) # y = (bs , 1 , hidden_size)

            if prevs[0] is None:
                prevs[0] = y
            else:
                prevs[0] = torch.cat([prevs[0] , y],dim = 1)

            for layer_i , decoder_layer_i in enumerate(self.decoders.__module__):

                prev = prevs[layer_i]   # (bs , t , hidden_size)

                _,y,_,_,_ = decoder_layer_i(encoder_result,y,decoder_pad_mask,None,prev)
                # (bs , 1 , hidden_size)

                if prevs[layer_i+1] is None:
                    prevs[layer_i+1] = y
                else:
                    prevs[layer_i+1] = torch.cat([prevs[layer_i+1],y],dim = 1)
                # (bs , t , hidden_size)

            y_hat_t = self.generator(y)    # (bs , 1 , d_output_size)
            indice = torch.argmax(torch.log_softmax(y_hat_t,dim = -1),dim = -1) # (bs , 1) indice

            indice = indice.masked_fill_(decoding_sum , pad_token)
            indices[:,t] = indice

            decoding_sum += (y_hat_t == eos_token)

            t += 1
            y = y_hat_t

        return indices


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
        self.encoders = coder_sequential(*encoder_layers)

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

import data

def decode_tgt(y_tensor,tgt_field): ### bs , n

    index_to_word_list = tgt_field.vocab.itos
    splited = []
    for sen in y_tensor:
        sentence = []
        sen = sen[sen != pad_token]
        for idx in sen:
            sentence.append(index_to_word_list[idx])
        splited.append(sentence)

    return splited

def test():
    wmt_loader = data.wmtloader('wmt16imsi.csv','wmt16_valid.csv',
                           max_length=255,
                            batch_size=16,
                            device=-1)
    input_size , output_size = len(wmt_loader.src.vocab) , len(wmt_loader.tgt.vocab)
    model = Transformer(input_size,output_size,
                                    layers=6,
                                    hidden_size=768,
                                    heads=8,
                                    dropout_p=0.1,
                                    max_length=256 ,
                                    first_norm=True)

    for i , mini_batch in enumerate(wmt_loader.train_iter):
        mini_batch.de = mini_batch.de[0]
        mini_batch.en = mini_batch.en[0]

        x, y = mini_batch.de, mini_batch.en[:, 1:]

        y_hat = model(x,mini_batch.en[:, :-1])
        y_result = torch.argmax(y_hat,dim=-1)
        print(i)
        print(y)
        print(y_result)

        pass


#test()