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
    
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self,min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD',
                           SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)