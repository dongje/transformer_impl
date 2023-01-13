import pandas as pd
from torch.utils.data import Dataset , DataLoader
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("wmt16" , 'de-en')

train_data = dataset['train']['translation']
valid_data = dataset['validation']['translation']
test_data = dataset['test']['translation']
datalist = [train_data,valid_data,test_data]

def split(dataset):
    de_data = []
    en_data = []

    for data1 in tqdm(dataset):
        de_data.append(data1['de'])
        en_data.append(data1['en'])

    return de_data,en_data

for i ,dataset in enumerate(datalist):
    de ,en = split(dataset)
    df = pd.DataFrame({'de' : de , 'en' : en})
    df.to_csv(f'wmt16_{i}.csv',encoding= 'utf-8',index=False)

