#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()

    for line in lines[:n]:
        print(line)


# printLines(os.path.join(corpus, 'movie_lines.txt'))

# 把文件中的每一行分隔成字典字段
def loadLines(filename, fields):
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            # 提取字段
            lineobj = {}
            for i, field in enumerate(fields):
                lineobj[field] = values[i]
            lines[lineobj['lineID']] = lineobj
    return lines


# 基于*movie_conversations.txt*将`loadLines`得到行字段分组到对话中
def loadConversations(filename, lines, fields):
    conversations = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            convobj = {}
            for i, field in enumerate(fields):
                convobj[field] = values[i]
            lineIds = eval(convobj['utteranceIDs'])
            convobj['lines'] = []
            for lineId in lineIds:
                convobj['lines'].append(lines[lineId])
            conversations.append(convobj)

    return conversations


# 从对话中提取句子
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines']) - 1):
            inputline = conversation['lines'][i]['text'].strip()
            targetline = conversation['lines'][i+1]['text'].strip()

            if inputline and targetline:
                qa_pairs.append([inputline, targetline])
    return qa_pairs


# 定义新生成文件的路径
datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

# 初始化行字典、对话列表和字段id
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVERSATIONS_FIELDS = ['character1ID',
                              'character2ID', 'movieID', 'utteranceIDs']

print('\nProcessing corpus...')
lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
print('\nLoading conversations...')
conversations = loadConversations(os.path.join(
    corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)

# 写入新文件
print('\nWriting newly formatted file...')
with open(datafile, 'w', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# # 输出部分样本
# print('\nSample lines from file:')
# printLines(datafile)

# 默认文字标记
PAD_token = 0   # 用来追加段句子
SOS_token = 0   # 句首标记
EOS_token = 0   # 句尾标记


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD',
                           SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(
                keep_words)/len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD',
                           SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


MAX_LENGTH = 10

# Unicode字符串转ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写化，裁剪、移除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 读取问/大对并返回词汇表对象
def readVoc(datafile, corpus_name):
    print('Reading lines...')
    # 读取文件并切割成行
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')

    # 分割行成对并标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# 如果问/答对p中的语句都在MAX_LENGTH阈值之下，返回True
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 过滤
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 使用上面定义的函数，返回填充过的词汇表和问答对列表
def loadPrepareDate(corpus,corpus_name,datafile,save_dir):
    print('Start preparing training data ...')
    voc,pairs = readVoc(datafile,corpus_name)
    print('Read {!s} sentence pairs'.format(len(pairs)))
    pairs = filterPairs(pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print('Counted words: ',voc.num_words)
    return voc,pairs

# 调用
save_dir = os.path.join('data','save')
voc,pairs = loadPrepareDate(corpus,corpus_name,datafile,save_dir)
print('\npairs:')
for pair in pairs[:10]:
    print(pair)  


MIN_COUNT = 3  

def trimRareWords(voc,pairs,MIN_COUNT):
    # 裁剪词汇表中使用次数在MIN_COUNT阈值之下的词
    voc.trim(MIN_COUNT)
    # 使用裁剪的词过滤语句对
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # 检查输入语句
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查输出语句
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print('Trimmed from {} pairs to {}, {:.4f} of total'.format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
    return keep_pairs

pairs =trimRareWords(voc,pairs,MIN_COUNT)