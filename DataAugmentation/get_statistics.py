import os
import json
import warnings
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from nltk.translate.bleu_score import sentence_bleu

warnings.filterwarnings("ignore")

DATA_DIR = '/lfs/local/0/paridhi/DialogueSystems/MinTL/damd_multiwoz'

ALL_AUGMENTATIONS = ['entity_replacement', 'crop', 'rotate', 'sequential', 'paraphrase_multi', 'translate']


def load_train_data(version, augmentation=None):
	if augmentation:
		filename = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_{augmentation}.json'
	else:
		filename = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd.json'

	with open(filename, 'r') as f:
		data = json.load(f, object_pairs_hook=OrderedDict)

	test_list = [l.strip().lower() for l in open(f'{DATA_DIR}/data/multi-woz-{version}/testListFile.json', 'r').readlines()]
	dev_list = [l.strip().lower() for l in open(f'{DATA_DIR}/data/multi-woz-{version}/valListFile.json', 'r').readlines()]

	dev_files, test_files = {}, {}
	for fn in test_list:
		test_files[fn.replace('.json', '')] = 1
	for fn in dev_list:
		dev_files[fn.replace('.json', '')] = 1

	data_train = {}
	for fn in list(data.keys()):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			data_train[fn] = data[fn]

	return data_train


def print_statistics(data, keys):
	print(f'# Dialogues: {len(keys)}')
	num_turns = [len(data[k]['log']) for k in keys]
	print(f'# Turns (total): {np.sum(num_turns)}')
	print(f'# Turns (average): {np.mean(num_turns):.3f}')
	user_utterance_lengths = [len(turn['user'].split()) for k in keys for turn in data[k]['log']]
	print(f'Utterance length: {np.mean(user_utterance_lengths):.3f}')


def get_bleu_score(data, keys):
	bleus = []
	for fn in keys:
		aug = data[fn]
		orig = data[fn.rstrip('_augment')]
		for idx in range(len(aug['log'])):
			orig_turn = orig['log'][idx]['user']
			aug_turn = aug['log'][idx]['user']
			curr = sentence_bleu(orig_turn.split(), aug_turn.split(), weights=(1, 1, 0, 0))
			bleus.append(curr)
	print(f'BLEU Score: {np.mean(bleus):.3f}')


if __name__=='__main__':

	data_orig = load_train_data('2.1')
	print('----- Training Statistics for Raw Data -----')
	print_statistics(data_orig, list(data_orig.keys()))

	for augmentation in ALL_AUGMENTATIONS:
		data_aug = load_train_data('2.1', augmentation)
		fns_augment = [x for x in list(data_aug.keys()) if x.endswith('_augment')]
		print(f'----- Training Statistics for Augmentation {augmentation} -----')
		print_statistics(data_aug, fns_augment)
		get_bleu_score(data_aug, fns_augment)
