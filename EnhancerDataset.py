import torch
from gensim.models import KeyedVectors
import numpy as np


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        if data_type not in ['train', 'validation', 'test']:
            raise ValueError('Invalid dataset type for EnhancerDataset object.')
        # 1. create the dna2vec object
        dna2vec_fp = "dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v"
        self.kv = KeyedVectors.load_word2vec_format(dna2vec_fp, binary=False) # use: kv['AGCA']
        
        if data_type == 'train':
            # contains {'seqs': sequences_array, 'labels': targets_array}
            self.data = torch.load('train_raw.pt')

    # get sample
    def __getitem__(self, idx):
        # file paths
        seq = self.data['seqs'][idx]
        target = self.data['labels'][idx]
        
        # split into 4-mers and encode using dna2vec
        tokens = []
        for i in range(len(seq)-4+1):
            kmer = seq[i:i+4]
            tokens.append(self.kv[kmer])
        # tokens.shape = (N, embed_dim)

        # convert to torch tensors
        tokens = torch.tensor(tokens, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return tokens, target

    def __len__(self):
        return len(self.data['seqs'])

