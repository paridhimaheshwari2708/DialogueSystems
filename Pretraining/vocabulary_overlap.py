'''
Download google n-grams dataset using
	google-ngram-downloader download --ngram-len 1 --output google_unigrams --verbose
'''

import os
import glob
import nltk
import string
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt


def clean_ngram(text):
	text = text.rstrip('_NOUN')
	text = text.rstrip('_VERB')
	text = text.rstrip('_DET')
	text = text.rstrip('_ADJ')
	text = text.rstrip('_ADV')
	text = text.rstrip('_ADP')
	text = text.rstrip('_CON')
	text = text.rstrip('_X')
	text = text.rstrip('_NUM')
	text = text.lower()
	return text


def get_ngrams_from_google(data_path, file_format, vocab_size):
	# all_files = glob.glob(os.path.join(data_path, file_format))
	# all_files = list(map(os.path.basename, all_files))

	# aggregate = pd.DataFrame()
	# for filename in tqdm(all_files):
	# 	df = pd.read_csv(
	# 		os.path.join(data_path, filename),
	# 		header=None,
	# 		names=['ngram', 'year', 'match_count', 'volume_count'],
	# 		delimiter='\t',
	# 	)
	# 	df = df.groupby('ngram').agg({'match_count': 'sum'}).reset_index()
	# 	df['ngram'] = df['ngram'].apply(clean_ngram)
	# 	df = df.groupby('ngram').agg({'match_count': 'sum'}).reset_index()
	# 	df = df.sort_values('match_count', ascending=False)
	# 	df = df.head(vocab_size)
	# 	aggregate = pd.concat([aggregate, df], axis=0)
	# aggregate = aggregate.sort_values('match_count', ascending=False)
	# aggregate.to_csv(os.path.join(data_path, 'aggregate.csv'))
	aggregate = pd.read_csv(os.path.join(data_path, 'aggregate.csv'), header=0, names=['ngram', 'match_count'])
	topk = aggregate[:vocab_size]['ngram'].tolist()
	return topk


def get_ngrams_from_file(data_path, ngram_length, vocab_size):
	with open(data_path, encoding="utf-8") as f:
		lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

	vocab = Counter()
	for line in tqdm(lines):
		token = nltk.word_tokenize(line)
		curr = ngrams(token, ngram_length)
		vocab.update(Counter(curr))

	topk = [x[0][0] for x in vocab.most_common(vocab_size)]
	return topk


# def plot_
if __name__=='__main__':

	K = 10000

	vocab_all = get_ngrams_from_file('all_data.txt', ngram_length=1, vocab_size=K)
	vocab_multiwoz = get_ngrams_from_file('multiwoz_data.txt', ngram_length=1, vocab_size=K)
	# vocab_multiwoz = get_ngrams_from_file('multiwoz_seq_data.txt', ngram_length=1, vocab_size=K)

	google_path = 'google_unigrams'
	file_format = 'googlebooks-eng-all-1gram-20120701-[a-z].gz'
	vocab_google = get_ngrams_from_google(google_path, file_format, vocab_size=K)

	labels = ['Google Books', 'DAPT', 'TAPT']
	datasets = [set(vocab_google), set(vocab_all), set(vocab_multiwoz)]
	num_labels = len(labels)

	overlaps = np.eye(num_labels) * 100
	for (i, j) in itertools.combinations_with_replacement(np.arange(num_labels), 2):
		if i == j:
			continue
		# curr = len(datasets[i].intersection(datasets[j])) / len(datasets[i].union(datasets[j])) * 100.0
		curr = len(datasets[i].intersection(datasets[j])) / K * 100.0
		overlaps[i,j] = curr
		overlaps[j,i] = curr

	plt.figure()
	sns.set(font_scale=1.2)
	sns.heatmap(overlaps, vmin=0.0, vmax=100.0, annot=True, fmt='.1f', xticklabels=labels, yticklabels=labels, cmap='Blues')
	plt.savefig('vocabulary_overlap.png')
