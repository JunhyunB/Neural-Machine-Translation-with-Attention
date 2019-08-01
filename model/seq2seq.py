import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hidden_size=1000, batch_size=80, embedding_size=620):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.encoder = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, embedded, hidden):
        output, hidden = self.encoder(embedded, hidden)

        return output, hidden

    def initHidden(self):
        return nn.init.orthogonal_(torch.empty(2, self.batch_size, self.hidden_size))

class Decoder(nn.Module):
    def __init__(self, hidden_size=1000, batch_size=80, embedding_size=620):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.decoder = nn.GRU(embedding_size, hidden_size, batch_first=True)
        
    def forward(self, embedded, hidden):
        output, hidden = self.decoder(embedded, hidden)

        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size=1000, vocab_len=None, embedding_size=620, batch_size=80, pad_idx=0):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.encoder = Encoder(hidden_size, batch_size)
        self.decoder = Decoder(hidden_size, batch_size)

        # src, trg share embedding
        self.embedding = nn.Embedding(vocab_len, embedding_size, padding_idx=pad_idx)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, vocab_len)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, trg):
        src_embedded = self.embedding(src)
        src_embedded = self.dropout(src_embedded)
        trg_embedded = self.embedding(trg)
        trg_embedded = self.dropout(trg_embedded)

        enc_init = self.encoder.initHidden()
        enc_output, enc_hidden = self.encoder(src_embedded, enc_init) # enc_output : (B, seq_len, hidden*2)  enc_hidden : (2, B, hidden_size)

        dec_output, _ = self.decoder(trg_embedded, enc_hidden[-1].unsqueeze(0)) # In the paper, they used backward hidden of enc.
        output = self.classifier(dec_output)
        output = self.softmax(output)

        return output