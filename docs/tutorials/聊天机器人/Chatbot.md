# 聊天机器人

来自[这里](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)。  

在本教程中，我们将探索一个有趣的用例，即循环序列到序列模型。我们将使用[Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)中的电影脚本来训练一个简单的聊天机器人。  

会话模型是人工智能研究的一个热门话题。聊天机器人可以在各种环境中找到，包括客户服务程序和在线帮助。这些机器人通常由基于检索的模型提供支持，这些模型输出对特定形式问题的预定义响应。在像公司的IT服务台这样高度受限的领域，这些模型可能够用了，但对于更一般的使用场景，它们还不够健壮。教会机器在多个领域与人进行有意义的对话是一个远未解决的问题。最近，深度学习热潮使得强大的生成模型成为可能，比如Google的神经会话模型，这标志着向多领域生成会话模型迈进了一大步。在本教程中，我们将使用PyTorch来实现这一模型。  

![chatbot](../../../pics/chatbot.png)  

```text
> hello?
Bot: hello .
> where am I?
Bot: you re in a hospital .
> who are you?
Bot: i m a lawyer .
> how are you doing?
Bot: i m fine .
> are you my friend?
Bot: no .
> you're under arrest
Bot: i m trying to help you !
> i'm just kidding
Bot: i m sorry .
> where are you from?
Bot: san francisco .
> it's time for me to leave
Bot: i know .
> goodbye
Bot: goodbye .
```
## 教程要点  

* 导入和处理[<font color=red>Cornell Movie-Dialogs Corpus</font>](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)数据集
* 使用[<font color=red>Luong attention mechanism(s)</font>](https://arxiv.org/abs/1508.04025)实现*序列到序列*模型
* 使用最小批量联合训练编码器和解码器
* 实现greedy-search解码模型
* 与训练有素的机器人互动  

## 准备工作

开始之前，需要先从[这里](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)下载数据包并解压到当前目录下的`data/`目录下。然后我们导入一些必要的包：  

```python
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
``` 

## 导入和处理数据  

接下来要做的是重新格式化我们的数据并导入到我们要使用的结构中。  

[<font color=red>Cornell Movie-Dialogs Corpus</font>](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)是一个丰富的电影任务对话数据集：  

* 10,292对电影人物间的220,579次对话交流
* 来自617部电影的9035位电影人物
* 总计304,713次对话

该数据集庞大多样，语言形式、时间、情感等方面存在较大差异。我们希望这种多样性使我们的模型对许多形式的输入和查询都具有健壮性。

首先，我们查看下数据文件中一些行，来了解下原始格式：  

```python
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()

    for line in lines[:n]:
        print(line)


printLines(os.path.join(corpus, 'movie_lines.txt'))
```  

输出：  

```text
b'L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!\n'
b'L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!\n'
b'L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.\n'
b'L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?\n'
b"L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.\n"
b'L924 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow\n'
b"L872 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Okay -- you're gonna need to learn how to lie.\n"
b'L871 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ No\n'
b'L870 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I\'m kidding.  You know how sometimes you just become this "persona"?  And you don\'t know how to quit?\n'
b'L869 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Like my fear of wearing pastels?\n'
```  

### 创建格式化的数据文件  

为了方便起见，我们将创建一个格式友好的数据文件，其中每一行都包含一个查询语句和响应语句对。  

下面的函数有助于解析原始的*movie_lines.txt*数据文件。  

* `loadLines`把文件中的每一行分隔成字典字段（行号、角色序号、电影序号、演员、文本等）
* `loadCinversations` 基于*movie_conversations.txt*将`loadLines`得到行字段分组到对话中
* `extractSentencePairs` 从对话中提取句子  

```python
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
```  

接下来我们调用这些函数来生成文件，命名为`formatted_movie_lines.txt`。  

```python
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

# 输出部分样本
print('\nSample lines from file:')
printLines(datafile)
```  

输出：  

```text
Processing corpus...

Loading conversations...

Writing newly formatted file...

Sample lines from file:
b"Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\tWell, I thought we'd start with pronunciation, if that's okay with you.\n"
b"Well, I thought we'd start with pronunciation, if that's okay with you.\tNot the hacking and gagging and spitting part.  Please.\n"
b"Not the hacking and gagging and spitting part.  Please.\tOkay... then how 'bout we try out some French cuisine.  Saturday?  Night?\n"
b"You're asking me out.  That's so cute. What's your name again?\tForget it.\n"
b"No, no, it's my fault -- we didn't have a proper introduction ---\tCameron.\n"
b"Cameron.\tThe thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\n"
b"The thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser.  My sister.  I can't date until she does.\tSeems like she could get a date easy enough...\n"
b'Why?\tUnsolved mystery.  She used to be really popular when she started high school, then it was just like she got sick of it or something.\n'
b"Unsolved mystery.  She used to be really popular when she started high school, then it was just like she got sick of it or something.\tThat's a shame.\n"
b'Gosh, if only we could find Kat a boyfriend...\tLet me see what I can do.\n'
```  

### 导入和整理数据  

我们接下来要做的是创建一个词汇表并将问/答语句对导入到内存中。  

注意，我们处理的是单词序列，它没有到离散数字空间的隐式映射。因此我们必须通过映射我们在数据集中遇到的每一个唯一的词到一个索引值的方法创建一个。  

所以我们定义一个`Voc`类，它保存词到索引的映射、索引到词的反映射、每个单词的计数和总词数。类中提供了向词汇表中添加单词(`addWord`)、添加一个句子中所有的词(`addSentence`)和修建不常见的词(`trim`)。更多关于`trimming`稍后介绍。  

```python

```

