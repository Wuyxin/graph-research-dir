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
        wordcloud = WordCloud(max_font_size=60, width=1200, height=700, background_color="white", colormap='Set2').generate(all_text)
        plt.figure(figsize=(12,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'figures/{year}.png', dpi=700)
        plt.cla()
    
    text_list = []
    for w in single_list:
        if w == 'sampling' or w == 'samples' or w == 'sample':
            text_list.append('sample')
            continue
        if 'generaliz' in w:
            text_list.append('generalization')
            continue
        for i_w in interest_list:
            if i_w == 'knowledge graph' and 'knowledgegraph' in w:
                text_list.append(i_w)
                break
            if i_w in w:
                text_list.append(i_w)
                break
    fdist = FreqDist(text_list)
    plt.figure(figsize=(10, 8))
    _dict = {i_w: fdist[i_w] for i_w in interest_list}
    
    index = np.argsort(-np.array(list(_dict.values())))
    print(np.array(list(_dict.values()))[index]/np.array(list(_dict.values()))[index].sum())
    plt.plot(np.array(list(_dict.values()))[index], np.array(list(_dict.keys()))[index], linewidth=2)
    plt.title(f'ICLR {year} - Frequency words in GNN research', weight='bold')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figures/{year}-dist.png', dpi=300)
    plt.cla()
    
    increase_index = np.argsort(np.array(list(_dict.values())))
    rank = np.zeros(len(interest_list))
    rank[increase_index] = np.arange(len(interest_list)) + 1
    values.append(rank)
    
# positive 
interest_list = np.array(interest_list)
idx = {'pos': (values[-1] >= values[0]) * (values[-1] >= values[1])  * (values[1] >= values[0]), 'neg': (values[-1] < values[0]) * (values[1] < values[0]) * (values[1] > values[-1])}
for key in ['pos', 'neg']:
    
    font = {'weight' : 'bold',
            'size'   : 40}
    matplotlib.rc('font', **font)
    idx_key = idx[key]
    barWidth = 0.25
    fig = plt.subplots(figsize =(30, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(interest_list[idx_key]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, values[0][idx_key], color ='r', width = barWidth,
            edgecolor ='grey', label ='2021')
    plt.bar(br2, values[1][idx_key], color ='g', width = barWidth,
            edgecolor ='grey', label ='2022')
    plt.bar(br3, values[2][idx_key], color ='b', width = barWidth,
            edgecolor ='grey', label ='2023')
    
    # Adding Xticks
    plt.ylabel('Inversed Rank', fontweight ='bold', fontsize = 40)
    plt.xticks([r + barWidth for r in range(len(interest_list[idx_key]))],
                interest_list[idx_key])

    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{key}_trend.png', dpi=300)

    plt.show()
    plt.cla()
