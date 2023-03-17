#!/usr/bin/env python
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""

import os
from os import path
import matplotlib
import matplotlib.pyplot as plt

from wordcloud import WordCloud
import nltk

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
nltk.download('punkt')

fdists = {}
interest_list = ['molecule', 'protein', 'transformer', 'causal', 'KG', 'sample', 'generalization',\
                 'contrastive', 'dynamic', 'spectral', 'distribution', 'domain', 'pretraining', 'architecture',\
                 'scale', 'attention', 'unsupervised', 'semisupervised', 'pretraining', 'robustness',\
                 'bias', 'explanation', 'generative', 'knowledgegraph']

font = {
        'weight' : 'bold',
        'size'   : 14,
        }

matplotlib.rc('font', **font)
for year in [2021, 2022, 2023]:

    # get data directory (using getcwd() is needed to support running example in generated IPython notebook)
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    # Read the whole text.
    all_text = open(path.join(d, f'abstracts/{year}.txt')).read().replace('-', '')
    all_text = all_text.replace('knowledge graph', 'knowledgegraph')
    lines = open(path.join(d, f'abstracts/{year}.txt')).readlines()
    text_list = [list(set(word_tokenize(line.lower().replace('-', '').replace('knowledge graph', 'knowledgegraph')))) for line in lines]
    single_list = []
    for _list in text_list:
        single_list.extend(_list)
    text = ' '.join([' '.join(_list) for _list in text_list])
    
    # lower max_font_size
    if False:
        wordcloud = WordCloud(max_font_size=60, width=1200, height=700).generate(all_text)
        plt.figure(figsize=(12,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'figures/{year}.png', dpi=500)
        plt.cla()
    
    text_list = []
    for w in single_list:
        if w == 'sampling' or w == 'samples' or w == 'sample':
            text_list.append('sample')
            continue
        for i_w in interest_list:
            if i_w in w:
                if i_w == 'knowledgegraph':
                    text_list.append('knowledge graph')
                else:
                    text_list.append(i_w)
                break
    fdist = FreqDist(text_list)
    plt.figure(figsize=(10,6))
    
    fdist.plot(30, title=f'ICLR {year} - Frequency words in GNN research')
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(f'figures/{year}-dist.png', dpi=300)
    plt.cla()
    fdists[year] = fdist

