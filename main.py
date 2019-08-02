import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as torchdata
from torch.utils.data import DataLoader

from dataloader import CustomDataset
from model.seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    with_cuda = True
    device_ids = None
    print("------- GPU Working -------")
    print("[Current GPU]:" + str(torch.cuda.get_device_name(0)))
else:
    with_cuda = False
    device_ids = None
    print("------- CPU Working -------")

data_loaded = CustomDataset(path='data/eng-fra.txt')
pad_idx = data_loaded.vocab_stoi['<pad>']

hidden_size = 1000
vocab_len = len(data_loaded.vocab_stoi)
embedding_size = 620
batch_size = 80

train_loader = torchdata.DataLoader(dataset=data_loaded,
                                    collate_fn=data_loaded.custom_collate_fn,
                                    batch_size=batch_size)

trg_max_seq_len = next(iter(train_loader))[1].size(1) - 1 # <s> is not included

epochs = 1
interval = 1
learning_rate = 5e-2

model = Seq2Seq(hidden_size=hidden_size, vocab_len=vocab_len, embedding_size=embedding_size,
                batch_size=batch_size, pad_idx=pad_idx, trg_max_seq_len=trg_max_seq_len, device=device)

model.to(device)

optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index=pad_idx)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model parameters : {count_parameters(model):,}')
print("Training START!")

phase = 'Train'

for epoch in range(epochs):
    epoch += 1
    if phase == 'Train':
        model.train(True)
    else:
        model.train(False)
    train_loss = 0
    for no, batch in enumerate(train_loader):
        src = batch[0].to(device)
        trg_ = batch[1].to(device)
        trg = trg_[:,:-1]
        trg_real = trg_[:,1:]

        output = model(src, trg)

        loss = criterion(output.transpose(1,2), trg_real)

        if phase == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
        train_loss += loss.item()

    epoch_loss = train_loss / len(data_loaded)

    if (epoch % interval == 0) or (epoch == 1):
        print("{} Loss: {:.4f}".format(phase, epoch_loss))