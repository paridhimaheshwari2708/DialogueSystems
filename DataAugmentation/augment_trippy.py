import os
import re
import sys
import json
import random
import argparse
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from utils import *

def load_data(version):
    with open(f'../data/MULTIWOZ{version}/train_dials.json', 'r') as f:
        train_data = json.load(f)

    with open(f'../data/MULTIWOZ{version}/dialogue_acts.json', 'r') as f:
        dialogue_dicts = json.load(f)

    return train_data, dialogue_dicts

def augment_data_paraphrase(version):
    train_data, dialogue_dicts = load_data(version)

    outdir = f'../data/Multiwoz{version}-paraphrase/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_count = len(train_data)
    train_aug_count = 0

    augmentation = Paraphrase()
    for fn in tqdm(list(train_data.keys())):
        dial = train_data[fn]
        dial_aug = deepcopy(dial)

        ex_name = fn.split('.')[0] ## gives the example name without json tag
        dst = dialogue_dicts[ex_name]
        dst_aug = deepcopy(dst)
        for turn in dial_aug['log']:
            if(turn['metadata']=={}):
                turn['text'] = augmentation.get_paraphrased_sentences(turn['text'], num_return_sequences=1) 
        train_data[fn + '_augment.json'] = dial_aug
        dialogue_dicts[ex_name + '_augment'] = dst_aug
        train_aug_count += 1

    print(len(train_data), len(dialogue_dicts))

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f)
    
    
    return

def augment_data_translate(version):
    train_data, dialogue_dicts = load_data(version)

    outdir = f'../data/Multiwoz{version}-translate/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_count = len(train_data)
    train_aug_count = 0

    augmentation = Translate()
    for fn in tqdm(list(train_data.keys())):
        dial = train_data[fn]
        dial_aug = deepcopy(dial)

        ex_name = fn.split('.')[0] ## gives the example name without json tag
        dst = dialogue_dicts[ex_name]
        dst_aug = deepcopy(dst)
        for turn in dial_aug['log']:
            if(turn['metadata']=={}): ### user utterance
                turn['text'] = augmentation.get_translated_sentences(turn['text'])     
        
        train_data[fn + '_augment.json'] = dial_aug
        dialogue_dicts[ex_name + '_augment'] = dst_aug
        train_aug_count += 1

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f)

    return

def augment_data_crop_rotate(version, operation):

    LENGTH_CONSTRAINT = 0.6
    train_data, dialogue_dicts = load_data(version)

    outdir = f'../data/Multiwoz{version}-{operation}/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_count = len(train_data)
    train_aug_count = 0

    stats = []

    augmentation = Crop_Rrotate()
    for fn in tqdm(list(train_data.keys())):
        dial = train_data[fn]
        dial_aug = deepcopy(dial)

        ex_name = fn.split('.')[0] ## gives the example name without json tag
        dst = dialogue_dicts[ex_name]
        dst_aug = deepcopy(dst)
        flag = False
        for turn in dial_aug['log']:
            if(turn['metadata']=={}):
                orig_sent = turn['text']
                orig_sent_len = len(orig_sent.split())
                translated_sent_list = augmentation.get_augmentation(orig_sent, operation)
                if(len(translated_sent_list) != 0):
                    translated_sent = translated_sent_list[0]
                    translated_sent_len = len(translated_sent.split())
                    stats.append((orig_sent_len, translated_sent_len))
                    if(translated_sent_len > (LENGTH_CONSTRAINT*orig_sent_len)):
                        turn['text'] = translated_sent    
                        flag = True
        if(flag == True):
            train_data[fn + '_augment.json'] = dial_aug
            dialogue_dicts[ex_name + '_augment'] = dst_aug
            train_aug_count += 1

    print(len(train_data), len(dialogue_dicts))

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f)

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	parser.add_argument("--version", type=str, help="MultiWOZ dataset version")
	args = parser.parse_args()

	if args.mode == 'paraphrase':
		augment_data_paraphrase(args.version)
	elif args.mode == 'translate':
		augment_data_translate(args.version)
	elif args.mode == 'rotate':
		augment_data_crop_rotate(args.version, 'rotate')
	elif args.mode == 'crop':
		augment_data_crop_rotate(args.version, 'crop')
	# elif args.mode == 'entity_replacement':
	# 	augment_data_entity_replacement(args.version)
	# elif args.mode == 'sequential':
	# 	augment_data_sequential(args.version)
