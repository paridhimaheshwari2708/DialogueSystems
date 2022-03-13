import os
import re
import sys
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from utils import *

DATA_DIR = '/lfs/local/0/paridhi/DialogueSystems/MinTL/damd_multiwoz'
LOGS_DIR = '/lfs/local/0/paridhi/DialogueSystems/MinTL/experiments_DST'

ALL_AUGMENTATIONS = ['paraphrase_multi', 'crop', 'rotate']

ENTITY_REPLACEMENT_PROBABILITY = 0.5

def load_data(version):
	with open(f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd.json', 'r') as f:
		data = json.load(f, object_pairs_hook=OrderedDict)

	test_list = [l.strip().lower() for l in open(f'{DATA_DIR}/data/multi-woz-{version}/testListFile.json', 'r').readlines()]
	dev_list = [l.strip().lower() for l in open(f'{DATA_DIR}/data/multi-woz-{version}/valListFile.json', 'r').readlines()]

	dev_files, test_files = {}, {}
	for fn in test_list:
		test_files[fn.replace('.json', '')] = 1
	for fn in dev_list:
		dev_files[fn.replace('.json', '')] = 1

	# with open(f'{DATA_DIR}/db/value_set_{version}_processed.json', 'r') as f:
	# 	value_set = json.load(f)

	with open(f'ontology_{version}.json', 'r') as f:
		value_set = json.load(f)

	return data, dev_files, test_files, value_set


def augment_data_paraphrase(version, multi):
	data, dev_files, test_files, _ = load_data(version)

	train_count_final = train_count = 0
	augmentation = Paraphrase()
	for fn in tqdm(list(data.keys())):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			dial = data[fn]
			dial_aug = deepcopy(dial)
			for turn in dial_aug['log']:
				if multi:
					out = augmentation.get_paraphrased_sentences(turn['user'], num_return_sequences=5)
					out = sorted(out, key=lambda x: len(x.split()), reverse=True)
				else:
					out = augmentation.get_paraphrased_sentences(turn['user'], num_return_sequences=1)
				turn['user'] = out[0]
			data[fn + '_augment'] = dial_aug
			train_count += 1
			train_count_final += 1

	print(f'# training data: {train_count}')
	print(f'# training data after augmentation: {train_count + train_count_final}')

	if multi:
		save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_paraphrase_multi.json'
	else:
		save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_paraphrase.json'
	with open(save_file, 'w') as f:
		json.dump(data, f)


def augment_data_translate(version):
	data, dev_files, test_files, _ = load_data(version)

	train_count_final = train_count = 0
	augmentation = Translate()
	for fn in tqdm(list(data.keys())):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			dial = data[fn]
			dial_aug = deepcopy(dial)
			for turn in dial_aug['log']:
				turn['user'] = augmentation.get_translated_sentences(turn['user'])
			data[fn + '_augment'] = dial_aug
			train_count += 1
			train_count_final += 1

	print(f'# training data: {train_count}')
	print(f'# training data after augmentation: {train_count + train_count_final}')

	save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_translate.json'
	with open(save_file, 'w') as f:
		json.dump(data, f)


def augment_data_crop_rotate(version, operation):
	LENGTH_CONSTRAINT = 0.6

	data, dev_files, test_files, _ = load_data(version)

	train_count_final = train_count = 0
	augmentation = Crop_Rotate()
	for fn in tqdm(list(data.keys())):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			dial = data[fn]
			dial_aug = deepcopy(dial)
			flag = False
			for turn in dial_aug['log']:
				orig_sent = turn['user']
				orig_sent_len = len(orig_sent.split())
				translated_sent = augmentation.get_augmentation(orig_sent, operation)
				if translated_sent !=  orig_sent:
					translated_sent_len = len(translated_sent.split())
					if translated_sent_len > (LENGTH_CONSTRAINT * orig_sent_len):
						turn['user'] = translated_sent
						flag = True
			train_count += 1
			if flag == True:
				data[fn + '_augment'] = dial_aug
				train_count_final += 1

	print(f'# training data: {train_count}')
	print(f'# training data after augmentation: {train_count + train_count_final}')

	save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_{operation}_multi.json'
	with open(save_file, 'w') as f:
		json.dump(data, f)


def augment_data_entity_replacement(version):

	def check_overlap(changes):
		changes = sorted(changes, key=lambda x: x[2])
		for index in range(1, len(changes)):
			if changes[index][2] <= changes[index - 1][3] - 1:
				return True
		return False

	data, dev_files, test_files, value_set = load_data(version)

	count1 = count2 = count3 = train_count_final = train_count = 0
	for fn in tqdm(list(data.keys())):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			dial = data[fn]
			train_count += 1

			all_turn_domains = set([turn['turn_domain'].replace('[', '').replace(']', '') for turn in dial['log']])
			all_turn_domains.discard('general')
			# No augmentation in these cases
			# -- No ontology for police
			# -- Multiple domains or something wrong with preprocessing for finding domain
			values_original = dial['log'][-1]['constraint_dict']
			if 'police' in all_turn_domains or not all_turn_domains.issubset(values_original):
				count1 += 1
				continue

			# Check if file is fine and should be augmented or not
			try:
				for turn in dial['log']:

					domain = turn['turn_domain'].replace('[', '').replace(']', '')
					if domain == 'general':
						continue

					user_orig = []
					curr_turn_values = turn['constraint_dict']
					for x in turn['user_delex'].split(' '):
						if not x.startswith('['):
							user_orig.append(x)
						else:
							slot = x.replace('[value_', '').replace(']', '')
							# Difference in naming conventions
							if slot not in curr_turn_values[domain]:
								if slot == 'price':
									slot = 'pricerange'
							# Trying slot in different domain of constraint dictionary
							if slot not in curr_turn_values[domain]:
								for d, k in curr_turn_values.items():
									if slot in k.keys():
										domain = d
										break
							assert slot in curr_turn_values[domain]
							user_orig.append(curr_turn_values[domain][slot])
					user_orig = ' '.join(user_orig)
					# assert user_orig == turn['user']
			except:
				count2 += 1
				continue

			# Ramdom sampling to replace values in the dialog
			v2v_dict = {}
			values_replaced = deepcopy(values_original)
			for domain, slot_value in values_original.items():
				for slot, value in slot_value.items():
					# Ignoring integer slots to avoid confusion
					if slot not in ['people', 'stay', 'stars']:
						if random.uniform(0, 1) > ENTITY_REPLACEMENT_PROBABILITY: #ontology4
							# picks = value_set[domain][slot]
							# picks = value_set[f'{domain}-{slot}'] # ontology
							# picks = [x for x in value_set[f'{domain}-{slot}'] if x != "do n't care" and x != "dontcare"] # ontology2
							picks = [x for x in value_set[f'{domain}-{slot}'] if x != "do n't care" and x != "dontcare" and "|" not in x] # ontology3
							values_replaced[domain][slot] = random.choice(picks)
						v2v_dict[values_original[domain][slot]] = values_replaced[domain][slot]

			# Iterating over dialogs and replacing
			dial_aug = deepcopy(dial)
			flag = False
			for turn in dial_aug['log']:

				domain = turn['turn_domain'].replace('[', '').replace(']', '')
				if domain == 'general':
					continue

				# Changes to be made - (1) 'user'
				user_new = []
				curr_turn_values = turn['constraint_dict']
				for x in turn['user_delex'].split(' '):
					if not x.startswith('['):
						user_new.append(x)
					else:
						slot = x.replace('[value_', '').replace(']', '')
						# Difference in naming conventions
						if slot not in curr_turn_values[domain]:
							if slot == 'price':
								slot = 'pricerange'
						# Trying slot in different domain of constraint dictionary
						if slot not in curr_turn_values[domain]:
							for d, k in curr_turn_values.items():
								if slot in k.keys():
									domain = d
									break
						assert slot in curr_turn_values[domain]
						user_new.append(values_replaced[domain][slot])
				turn['user'] = ' '.join(user_new)

				# Changes to be made - (2) 'resp_nodelex'
				changes = []
				for v_orig, v_new in v2v_dict.items():
					changes.extend([ (v_orig, v_new, m.start(), m.end()) for m in re.finditer(v_orig, turn['resp_nodelex'])])
				if check_overlap(changes):
					flag = True
					count3 += 1
					break
				changes = sorted(changes, key = lambda x: x[3], reverse=True)
				for (v_orig, v_new, start, end) in changes:
					turn['resp_nodelex'] = turn['resp_nodelex'][:start] + v_new + turn['resp_nodelex'][end:]

				# Changes to be made - (3) 'constraint'
				constraints = []
				for domain, info_slots in turn['constraint_dict'].items():
					constraints.append('['+domain+']')
					for slot in info_slots:
						constraints.append(slot)
						constraints.extend(values_replaced[domain][slot].split())
				turn['constraint'] = ' '.join(constraints)

			if not flag:
				train_count_final += 1
				data[fn + '_augment'] = dial_aug

	print(f'# training data: {train_count}')
	print(f'# training data after augmentation: {train_count + train_count_final}')
	print(f'Error:\n\tPolice or multi domain: {count1}\
					\n\tUser delexical and reconstructed mismatch: {count2}\
					\n\tResponse overlap fail: {count3}')

	save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_entity_replacement_ontology4.json'
	with open(save_file, 'w') as f:
		json.dump(data, f)


def augment_data_sequential(version, num_sequence=3):
	data, dev_files, test_files, _ = load_data(version)

	train_count_final = train_count = 0
	augmentation = Paraphrase()
	for fn in tqdm(list(data.keys())):
		if (not dev_files.get(fn)) and (not test_files.get(fn)):
			dial = data[fn]
			num_turns = len(dial['log'])
			dial_aug = {'log' : []}
			for i in range(num_turns - num_sequence + 1):
				curr_turn = {}
				curr_turn['user'] = ' '.join([x['user'] for x in dial['log'][i : i + num_sequence]])
				curr_turn['resp_nodelex'] = ' '.join([x['resp_nodelex'] for x in dial['log'][i : i + num_sequence]])
				curr_turn['constraint'] = dial['log'][i]['constraint']
				dial_aug['log'].append(curr_turn)
			data[fn + '_augment'] = dial_aug
			train_count += 1
			train_count_final += 1

	print(f'# training data: {train_count}')
	print(f'# training data after augmentation: {train_count + train_count_final}')

	save_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_sequential.json'
	with open(save_file, 'w') as f:
		json.dump(data, f)


def divide_data_quartiles(mode, version):
	data, dev_files, test_files, _ = load_data(version)

	load_file = f'{DATA_DIR}/data/multi-woz-{version}-processed/data_for_damd_{mode}.json'
	with open(load_file, 'r') as f:
		data_aug = json.load(f)

	not_train_files = list(dev_files.keys()) + list(test_files.keys())
	train_files = list(set(data.keys()).difference(not_train_files))
	train_aug_files = [x for x in data_aug.keys() if x.endswith('_augment')]

	# Dividing into quartiles
	random.shuffle(train_aug_files)
	q1, q2, q3, q4 = np.array_split(train_aug_files, 4)
	data_aug_quarter = deepcopy(data)

	# Adding 25 % of augmented data
	for fn in tqdm(q1):
		data_aug_quarter[fn] = data_aug[fn]
	with open(load_file.replace('.json', '_25.json'), 'w') as f:
		json.dump(data_aug_quarter, f)

	# Adding 50 % of augmented data
	for fn in tqdm(q2):
		data_aug_quarter[fn] = data_aug[fn]
	with open(load_file.replace('.json', '_50.json'), 'w') as f:
		json.dump(data_aug_quarter, f)

	# Adding 75 % of augmented data
	for fn in tqdm(q3):
		data_aug_quarter[fn] = data_aug[fn]
	with open(load_file.replace('.json', '_75.json'), 'w') as f:
		json.dump(data_aug_quarter, f)


def fetch_goal_accuracy(exp_name):
	with open(os.path.join(LOGS_DIR, exp_name, 'result.txt'), 'r') as f:
		content = f.readlines()
	acc = float(content[0].split()[4])
	return acc


def analyse_quartile_results(version):
	results = {}
	for mode in ALL_AUGMENTATIONS:
		if 'paraphrase' in mode:
			results[mode.rstrip('_multi')] = [
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs16_sp5_dc0.8_cw3_data_data_for_damd_{mode}_25_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs16_sp5_dc0.8_cw3_data_data_for_damd_{mode}_50_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs16_sp5_dc0.8_cw3_data_data_for_damd_{mode}_75_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs16_sp5_dc0.8_cw3_data_data_for_damd_{mode}_model_t5-small_noupdateFalse'),
			]
		else:
			results[mode] = [
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_data_data_for_damd_{mode}_25_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_data_data_for_damd_{mode}_50_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_data_data_for_damd_{mode}_75_model_t5-small_noupdateFalse'),
				fetch_goal_accuracy(f'v{version}_all_sd557_lr0.0006_bs32_sp5_dc0.8_cw3_data_data_for_damd_{mode}_model_t5-small_noupdateFalse'),
			]
	plot_quartile_results(results, [0, 25, 50, 75, 100], 'augmentation_rate_mintl.png')


if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	parser.add_argument("--version", type=str, help="MultiWOZ dataset version")
	args = parser.parse_args()

	if args.mode == 'plot':
		analyse_quartile_results(args.version)
	else:
		if args.mode == 'paraphrase':
			augment_data_paraphrase(args.version, multi=False)
		if args.mode == 'paraphrase_multi':
			augment_data_paraphrase(args.version, multi=True)
		elif args.mode == 'translate':
			augment_data_translate(args.version)
		elif args.mode == 'rotate':
			augment_data_crop_rotate(args.version, 'rotate')
		elif args.mode == 'crop':
			augment_data_crop_rotate(args.version, 'crop')
		elif args.mode == 'entity_replacement':
			augment_data_entity_replacement(args.version)
		elif args.mode == 'sequential':
			augment_data_sequential(args.version)
		divide_data_quartiles(args.mode, args.version)
