import pandas as pd
import torch

df = pd.read_csv('ungener1.csv', header=None)
print(df)

reverse_vocab = torch.load('D:/论文/magcn-seqgan/seq-gan/reverse_vocab_unlabel.pkl')
words_all = []
words = [0 for j in range(df.shape[1])]
print(words)
for i in range(df.shape[0]):
    words_all.append(words)
words_all = pd.DataFrame(words_all)
print(words_all)
count = 0
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        words_all.iloc[i, j] = reverse_vocab[df.iloc[i, j]]
    count = count + 1
    print(count)
print(words_all)
words_all.to_csv('unfword.csv', header=None, index=None)
