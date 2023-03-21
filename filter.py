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

import numpy as np 
from wordcloud import WordCloud

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
nltk.download('punkt')

fdists = {}
values = []
interest_list = ['molecule', 'protein', 'transformer', 'causal', 'sample', 'generalization',\
                 'contrastive', 'dynamic', 'spectral', 'distribution', 'pretrain', 'architecture',\
                 'scale', 'attention', 'unsupervised', 'semisupervised', 'robustness',\
                 'explanation', 'generative', 'knowledge graph', 'reinforcement']

font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)
filtere_word = 'protein'
for year in [2021, 2022, 2023]:

    # get data directory (using getcwd() is needed to support running example in generated IPython notebook)
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    # Read the whole text.
    all_text = open(path.join(d, f'abstracts/{year}.txt')).read().replace('-', '')
    all_text = all_text.replace('knowledge graph', 'knowledgegraph')
    lines = open(path.join(d, f'abstracts/{year}.txt')).readlines()
    text_list = [list(set(word_tokenize(line.lower().replace('-', '').replace('knowledge graph', 'knowledgegraph')))) for line in lines]
    filtered = [line for line in lines if filtere_word in line]
    filtered_test = " ".join(filtered)
    single_list = []
    for _list in text_list:
        single_list.extend(_list)
    text = ' '.join([' '.join(_list) for _list in text_list])
    
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=60, width=1200, height=700, background_color="white", colormap='Set2').generate(filtered_test)
    plt.figure(figsize=(12,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'figures/{year}-{filtere_word}.png', dpi=700)
    plt.cla()
    