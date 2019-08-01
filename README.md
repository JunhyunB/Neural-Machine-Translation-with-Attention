# Introduction
## Neural Machine Translation with Attention
This is PyTorch implementation for the paper   
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)  

# Requirements
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.5
* [torchtext](https://torchtext.readthedocs.io/en/latest/#) version >= 0.4.0

# Dataset
You need parallel text dataset in the data directory as shown below.   
Parallel language pair is splitted with TAB.  

```
You're just tired.	Tu es juste fatigué.
Everybody's in bed.	Tout le monde est au lit.
I didn't promise anybody anything.	Je n'ai rien promis à quiconque.
```
After prepare your own dataset, you should make tokenizer suitable for your dataset.  
I used [Google SentencePiece](https://github.com/google/sentencepiece) tokenizer.
```
pip install sentencepiece
git clone https://github.com/JunhyunB/Neural-Machine-Translation-with-Attention.git
cd data
python tokenizer.py
```
Before run tokenizer.py, you should change your data directory and target vocabulary size.  
( change data.txt and vocab_size 10000 for your dataset )
```
[tokenizer.py]
import sentencepiece as spm

s = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train('--input=data.txt --model_prefix=tokenizer --vocab_size=10000')
```