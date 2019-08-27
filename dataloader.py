import torch
import torch.utils.data as torchdata

from collections import defaultdict
from torchtext import data


# This is Google SentencePiece Tokenizer
# https://github.com/google/sentencepiece
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('data/tokenizer.model')

class CustomDataset(torchdata.Dataset):
    def __init__(self, path='data/eng-fra.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            source = []
            target = []
            while True:
                line = f.readline().strip().split('\t')
                if len(line) != 2:
                    break
                else:
                    source.append(sp.EncodeAsPieces(line[0]))
                    target.append(['<sos>']+sp.EncodeAsPieces(line[1])+['<eos>'])
        
        print("Tokenization Complete")
        print("Making Vocabulary...")

        flatten = lambda x: [tkn for s in x for tkn in s]
        unique_tokens = sorted(set(flatten(source+target)))
        
        self.vocab_stoi = defaultdict()
        self.vocab_stoi['<pad>'] = 0
        self.vocab_stoi['<unk>'] = 1

        for i, token in enumerate(unique_tokens, 2):
            self.vocab_stoi[token] = i

        self.vocab_itos = [t for t, i in sorted([(token, index) for token, index in self.vocab_stoi.items()], key=lambda x:x[1])]
        source_numerical = [list(map(self.vocab_stoi.get, s)) for s in source]
        target_numerical = [list(map(self.vocab_stoi.get, s)) for s in target]

        self.src = source_numerical
        self.trg = target_numerical
        self.pad_idx = self.vocab_stoi['<pad>']

        self.src_max_len = max(len(s) for s in self.src)
        self.trg_max_len = max(len(s) for s in self.trg)

    def __getitem__(self, index):
        # return index data
        return [self.src[index], self.trg[index]]

    def __len__(self):
        # length of data
        return len(self.src)
    
    def custom_collate_fn(self, data):
        src_, trg_ = list(zip(*data))

        SOURCE = [s+[self.pad_idx]*(self.src_max_len-len(s)) if len(s) < self.src_max_len else s[:self.src_max_len] for s in src_]
        TARGET = [s+[self.pad_idx]*(self.trg_max_len-len(s)) if len(s) < self.trg_max_len else s[:self.trg_max_len] for s in trg_]

        '''print(SOURCE)
        print(TARGET)'''
        
        return torch.LongTensor(SOURCE), torch.LongTensor(TARGET)