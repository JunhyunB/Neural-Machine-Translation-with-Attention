import sentencepiece as spm

s = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train('--input=dataset.txt --model_prefix=tokenizer --vocab_size=10000')