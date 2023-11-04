import os
import music21 as ms21
from collections import Counter
import torch
from prepare_data import filepath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
train_file = filepath + '/midi_train.txt'
test_file = filepath + '/midi_test.txt'

BOS = "BOS"  # begin of sentence
EOS = "EOS"  # end of sentence
UNK = "UNK"
PAD = "PAD"

PAD_IDX = 0
UNK_IDX = 1

def load_data(in_file):
  
    cn = []
    en = []
    with open(in_file, 'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip(' ').split('\t')
            en.append([BOS] + parts[0].split()+ [EOS])
            cn.append([BOS] + parts[1].split() + [EOS])
    return en, cn

train_en, train_cn = load_data(train_file)

def build_dict(sentences, max_words=500):
    
    counter = Counter()
    for sentence in sentences:
        for word in sentence:
            counter[word] += 1
    topn = counter.most_common(max_words)
    total_words = len(topn) + 2
    word_dict = {word[0]: i + 2 for i, word in enumerate(topn)}
    word_dict[PAD] = PAD_IDX
    word_dict[UNK] = UNK_IDX
    return word_dict, total_words

# word -> index
en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)

cn_bos_idx = cn_dict[BOS]
cn_eos_idx = cn_dict[EOS]

print(f"melody vocabulary size:{en_total_words}")
print(f"chord vocabulary size:{cn_total_words}")

# index -> index
en_dict_rev = {v: k for k, v in en_dict.items()}
cn_dict_rev = {v: k for k, v in cn_dict.items()}


def encode_sentences(sents, word_dict: dict):
    
    return [[word_dict.get(w, UNK_IDX) for w in s] for s in sents]


def decode_sentences(sents, word_dict_rev: dict):
    
    sents = sents.numpy()
    return [[word_dict_rev.get(w, UNK) for w in s] for s in sents]


def sort_sentences(en_sents, cn_sents):
    
    idx = sorted(range(len(en_sents)), key=lambda x: len(en_sents[x]))
    return [en_sents[i] for i in idx], [cn_sents[i] for i in idx]


class LanguageLoader:

    def __init__(self, file: str, batch_size=40, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_en, self.train_cn = load_data(file)
        self.sents_en = encode_sentences(self.train_en, en_dict)
        self.sents_cn = encode_sentences(self.train_cn, cn_dict)
        self.sents_en_lens = [len(v) for v in self.sents_en]
        self.sents_cn_lens = [len(v) for v in self.sents_cn]
        self.sents_en_lens_max = max(self.sents_en_lens)
        self.sents_cn_lens_max = max(self.sents_cn_lens)
        self._batch_index = 0
        self.batch_count = len(self.sents_en) // self.batch_size

    # padding
    def pad_sentences(self, sentences):
        lens = torch.LongTensor([len(s) for s in sentences])
        max_len = torch.max(lens)
        result = torch.zeros([lens.size(0), max_len], dtype=torch.long)
        for i, sentence in enumerate(sentences):
            result[i, :lens[i]] = torch.IntTensor(sentence)
        return result, lens

    def get_batch(self, i: int):
        s = i * self.batch_size
        e = (i + 1) * self.batch_size
        x_batch, x_lens = self.pad_sentences(self.sents_en[s:e])
        y_batch, y_lens = self.pad_sentences(self.sents_cn[s:e])
        x_batch, x_lens, y_batch, y_lens = x_batch.to(device), x_lens.to(device), y_batch.to(device), y_lens.to(device)
        return x_batch, x_lens, y_batch, y_lens

    def __len__(self):
        return self.batch_count

    def __next__(self):
        if self._batch_index > self.batch_count:
            raise StopIteration()
        r = self.get_batch(self._batch_index)
        self._batch_index += 1

        return r

    def __iter__(self):
        self._batch_index = 0
        return self


train_dataloader = LanguageLoader(train_file, batch_size=32)
test_dataloader = LanguageLoader(test_file, batch_size=32)


def decode_sents(sentences, is_cn=True):
   
    word_dict_rev = cn_dict_rev if is_cn else en_dict_rev
    r = decode_sentences(sentences, word_dict_rev=word_dict_rev)
    decoded_sents = []
    for v in r:
        sent = []
        for x in v:
            if x == EOS:
                break
            if x in [BOS, PAD]:
                continue
            sent.append(x)
        if is_cn:
            decoded_sents.append("".join(sent))
        else:
            decoded_sents.append(" ".join(sent))
    return decoded_sents


def answer_sents(sentences, is_cn=True):
    
    word_dict_rev = cn_dict_rev if is_cn else en_dict_rev
    r = decode_sentences(sentences, word_dict_rev=word_dict_rev)
    decoded_sents = []
    for v in r:
        sent = []
        for x in v:
            if x == EOS:
                break
            if x in [BOS, PAD]:
                continue
            sent.append(x)
        if is_cn:
            decoded_sents.append(" ".join(sent))
        else:
            decoded_sents.append(" ".join(sent))
    return decoded_sents


if __name__ == '__main__':
    
    print(len(train_dataloader))
