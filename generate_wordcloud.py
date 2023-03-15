#!/usr/bin/env python
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""

import os
from os import path
import matplotlib.pyplot as plt

from wordcloud import WordCloud

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
text = open(path.join(d, 'abstracts.txt')).read()

# Generate a word cloud image
# wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=60, width=1200, height=700).generate(text)
plt.figure(figsize=(12,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig('abstract.png', dpi=1000)
plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()