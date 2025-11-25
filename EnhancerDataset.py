import torch
from gensim.models import KeyedVectors
import numpy as np

def enhancer_collate_fn(batch):
    """This pads the data to the max sequence length in the batch to reduce memory usage during training
        Returns the sequences as padded, mask tensor for attention parameter, and labels
    """
    sequences, labels = zip(*batch)  # batch is list of (tokens, target)

    # First normalize all seqs to (L_i, embed_dim)
    arrays = []
    lengths = []
    for seq in sequences:
        arr = np.array(seq, dtype=np.float32)
        if arr.ndim == 1:
            # Treat as a single-token sequence: (embed_dim,) -> (1, embed_dim)
            arr = arr[None, :]
        arrays.append(arr)
        lengths.append(arr.shape[0])

    max_len = max(lengths)
    embed_dim = arrays[0].shape[1]

    padded = np.zeros((len(arrays), max_len, embed_dim), dtype=np.float32)
    mask = np.zeros((len(arrays), max_len), dtype=np.bool_)

    for i, arr in enumerate(arrays):
        L = arr.shape[0]
        padded[i, :L, :] = arr
        mask[i, :L] = 1  # 1 for real token, 0 for padding

    padded = torch.from_numpy(padded) # (batch, max_len, embed_dim)
    mask = torch.from_numpy(mask) # (batch, max_len)
    labels = torch.tensor(labels, dtype=torch.float32)  # (batch,)

    return padded, mask, labels


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        if data_type not in ['train', 'validation', 'test']:
            raise ValueError('Invalid dataset type for EnhancerDataset object.')
        
        # create the dna2vec object for O(1) kmer embeddings
        dna2vec_fp = "dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v"
        self.kv = KeyedVectors.load_word2vec_format(dna2vec_fp, binary=False) # use: kv['AGCA']
        
        if data_type == 'train':
            # contains {'seqs': sequences_array, 'labels': targets_array}
            self.data = torch.load('train_raw.pt')
        elif data_type == 'test':
            self.data = torch.load('test_raw.pt')
        else:
            print('validation set not done yet')

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
        if len(tokens) == 0:
            raise ValueError(f'Length of tokens array is 0 for index {idx}. Sequences array: ', seq)


        # tokens.shape = (N, embed_dim)

        # convert to torch tensors
        tokens = torch.tensor(tokens, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return tokens, target

    def __len__(self):
        return len(self.data['seqs'])

