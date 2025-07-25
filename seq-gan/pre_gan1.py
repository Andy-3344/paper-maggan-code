# -*- coding: utf-8 -*-
import pandas as pd
from itertools import chain
from collections import Counter
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('D:\论文\magcn-seqgan\dataset\lncrnadisease2.0\data result\data_unlabel_feature.csv', header=None)
print(df)
SEQ_LENGTH = df.shape[1]


def read_sampleFile(file='D:\论文\magcn-seqgan\dataset\lncrnadisease2.0\data result\data_unlabel_feature.csv', pad_token='PAD', num=None):
    if file[-3:] == 'pkl' or file[-3:] == 'csv':
        if file[-3:] == 'pkl':
            data = pd.read_pickle(file)
        else:
            data = pd.read_csv(file, header=None)

        if num is not None:
            num = min(num, len(data))
            data = data[0:num]
        lineList_all = data.values.tolist()
        characters = set(chain.from_iterable(lineList_all))
        x_lengths = [len(x) - Counter(x)[pad_token] for x in lineList_all]
    else:
        lineList_all = list()
        characters = list()
        x_lengths = list()
        count = 0
        with open(file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line.strip()
                lineList = list(line)
                try:
                    lineList.remove('\n')
                except ValueError:
                    pass
                x_lengths.append(len(lineList) + 1)
                characters.extend(lineList)
                if len(lineList) < SEQ_LENGTH:
                    lineList.extend([pad_token] * (SEQ_LENGTH - len(lineList)))
                lineList_all.append(['START'] + lineList)
                count += 1
                if num is not None and count >= num:
                    break

    vocabulary = dict([(y, x) for x, y in enumerate(set(characters))])
    reverse_vocab = dict([(x, y) for x, y in enumerate(set(characters))])

    # tmp = sorted(zip(x_lengths, lineList_all), reverse=True)
    # x_lengths = [x for x, y in tmp]
    # lineList_all = [y for x, y in tmp]
    generated_data = [vocabulary[x] for y in lineList_all for i, x in enumerate(y) if i < SEQ_LENGTH]
    x = torch.tensor(generated_data, device=DEVICE).view(-1, SEQ_LENGTH)
    return x.int(), vocabulary, reverse_vocab, x_lengths


x, vocabulary, reverse_vocab, x_lengths = read_sampleFile(num=1216)
x = x.tolist()
output_file = 'unlabel1.data'
with open(output_file, 'w') as fout:
    for sample in x:
        string = ' '.join([str(s) for s in sample])
        fout.write('%s\n' % string)
fout.close()
print(len(reverse_vocab))
torch.save(reverse_vocab, 'reverse_vocab_unlabel1.pkl')
