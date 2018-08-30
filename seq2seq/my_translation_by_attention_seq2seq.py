from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import os
import re
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import seaborn as sns
import torch.nn as nn
import visdom
from nltk.translate.bleu_score import sentence_bleu
from torch import optim
import torch.nn.functional as func
from torch.utils.data import DataLoader

from config import VISDOM_SERVER, VISDOM_PORT
from seq2seq import EncoderSeq, DecoderSeq, Seq2SeqTrainer
from snippets.scaffold import TrainLoop, TestLoop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


teacher_forcing_ratio = 0.5


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env="Eng-Fra NMT")
    vis.matplot(fig,
                opts=dict(title="seq2seq"),
                win="seq2seq")
    plt.close("all")


def evaluate(_encoder, _decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = None
        for ei in range(input_length):
            encoder_output, encoder_hidden = _encoder(input_tensor[ei],
                                                      encoder_hidden)
        decoder_input = torch.tensor([[SOS_token]], device=torch.cuda.current_device())  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoded_outputs = _decoder.forward_n(decoder_input, decoder_hidden, n_steps=max_length, stop_token=EOS_token)
        for output in decoded_outputs:
            _, top_index = output.topk(1)
            decoded_words.append(output_lang.index2word[top_index.item()])
        return decoded_words


hidden_size = 256
learning_rate = 1e-2

encoder = EncoderSeq(input_size=input_lang.n_words, hidden_size=hidden_size).cuda()
decoder = DecoderSeq(output_size=output_lang.n_words, hidden_size=hidden_size).cuda()

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

train_set_split = 0.5
training_pairs = [tensorsFromPair(pairs[i])
                  for i in range(int(len(pairs) * train_set_split))]
train_data_loader = DataLoader(dataset=training_pairs, batch_size=1, shuffle=False)
trainer = Seq2SeqTrainer(encoder=encoder, decoder=decoder,
                         optim_e=encoder_optimizer, optim_d=decoder_optimizer,
                         device=torch.cuda.current_device(),
                         eos_token=EOS_token, sos_token=SOS_token,
                         teach_forcing_prob=0.5,
                         criterion=func.nll_loss)
vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env="Eng-Fra NMT")

with TrainLoop(max_epochs=50, use_cuda=True).with_context() as loop:
    for epoch in loop.iter_epochs():
        for step, (input_seq, target_seq) in loop.iter_steps(train_data_loader):
            loss = trainer.step(input_seq.squeeze(0), target_seq.squeeze(0))
            loop.submit_metric("train_loss", loss)
        epochs, train_loss = loop.get_metric_by_name("train_loss")

        vis.line(X=np.asarray(epochs)[-1:], Y=np.sum(train_loss, -1)[-1:],
                 opts=dict(legend=["Train Loss"], title="Loss"),
                 win="seq2seq training", update="append" if epoch > 1 else False, )

torch.save(encoder.state_dict(), os.path.expanduser("~/experiments/Imp/seq2seq/seq2seq_encoder_state_dict.pkl"))
torch.save(decoder.state_dict(), os.path.expanduser("~/experiments/Imp/seq2seq/seq2seq_decoder_state_dict.pkl"))

test_pairs = pairs[len(training_pairs):]
with TestLoop(use_cuda=True).with_context() as test_loop:
    for _, (x, y) in test_loop.iter_steps(test_pairs[:100]):
        # print('>', x)
        # print('=', y)
        output_words = evaluate(encoder, decoder, x)
        output_sentence = ' '.join(output_words)
        test_loop.submit_metric("BLEU", sentence_bleu([y.split(" ")], output_sentence.split(" ")))
        # print('<', output_sentence)
        # print('')

fig = plt.figure(figsize=(5, 3), dpi=326)
sns.distplot(test_loop.get_metric_by_name("BLEU"),
             hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
vis.matplot(fig, opts=dict(title="BLEU dist"), win="seq2seq evaluation")
