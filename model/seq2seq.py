import torch
import torch.nn as nn
import torch.nn.utils.rnn as R

class Encoder(nn.Module):
    def __init__(self, hidden_size, batch_size, embedding_size, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device

        self.encoder = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, embedded, hidden):
        output, hidden = self.encoder(embedded, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        return nn.init.orthogonal_(torch.empty(2, batch_size, self.hidden_size)).to(self.device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, batch_size, embedding_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.decoder = nn.GRU(embedding_size, hidden_size, batch_first=True)
        
    def forward(self, embedded, hidden):
        output, hidden = self.decoder(embedded, hidden)

        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size=1000, vocab_len=None, embedding_size=620, batch_size=80, pad_idx=0, trg_max_seq_len=None, device=None):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.trg_max_seq_len = trg_max_seq_len
        self.device = device

        self.encoder = Encoder(hidden_size, batch_size, embedding_size, device)
        self.decoder = Decoder(hidden_size, batch_size, embedding_size)
        #self.alignment_model= nn.Linear(hidden_size*3,1)

        # src, trg share embedding
        self.embedding = nn.Embedding(vocab_len, embedding_size, padding_idx=pad_idx)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, vocab_len)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, trg):
        src_lengths = torch.LongTensor([torch.max(src[i,:].data.nonzero())+1 for i in range(src.size(0))])
        trg_lengths = torch.LongTensor([torch.max(trg[i,:].data.nonzero())+1 for i in range(trg.size(0))])
        src_embedded = self.embedding(src)
        src_embedded = self.dropout(src_embedded)
        src_embedded = R.pack_padded_sequence(src_embedded, src_lengths, batch_first=True, enforce_sorted=False)
        trg_embedded = self.embedding(trg)
        trg_embedded = self.dropout(trg_embedded)
        trg_embedded = R.pack_padded_sequence(trg_embedded, trg_lengths, batch_first=True, enforce_sorted=False)

        enc_init = self.encoder.initHidden(src.size(0))
        enc_output, enc_hidden = self.encoder(src_embedded, enc_init) # enc_output : (B, seq_len, hidden*2)  enc_hidden : (2, B, hidden_size)
        enc_output, _ = R.pad_packed_sequence(enc_output, batch_first=True, padding_value=0)
        dec_output, _ = self.decoder(trg_embedded, enc_hidden[-1].unsqueeze(0)) # In the paper, they used backward hidden of enc.
        dec_output, _ = R.pad_packed_sequence(dec_output, batch_first=True, padding_value=0)

        output = self.classifier(dec_output)
        output = self.softmax(output)

        padding_tensor = torch.zeros(output.size(0), self.trg_max_seq_len-output.size(1),output.size(2)).to(self.device)
        output = torch.cat((output, padding_tensor), dim=1)

        return output