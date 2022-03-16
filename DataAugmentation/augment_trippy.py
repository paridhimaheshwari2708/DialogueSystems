from genericpath import exists
import re
import sys
import json
import random
import argparse
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import os
import pickle
from utils import *
from crop_rotate import *

slot_maps = {'attraction-semi-area': 'attraction-area',
 'attraction-semi-name': 'attraction-name',
 'attraction-semi-type': 'attraction-type',
 'hotel-book-day': 'hotel-book_day',
 'hotel-book-people': 'hotel-book_people',
 'hotel-book-stay': 'hotel-book_stay',
 'hotel-semi-area': 'hotel-area',
 'hotel-semi-internet': 'hotel-internet',
 'hotel-semi-name': 'hotel-name',
 'hotel-semi-parking': 'hotel-parking',
 'hotel-semi-pricerange': 'hotel-pricerange',
 'hotel-semi-stars': 'hotel-stars',
 'hotel-semi-type': 'hotel-type',
 'restaurant-book-day': 'restaurant-book_day',
 'restaurant-book-people': 'restaurant-book_people',
 'restaurant-book-time': 'restaurant-book_time',
 'restaurant-semi-area': 'restaurant-area',
 'restaurant-semi-food': 'restaurant-food',
 'restaurant-semi-name': 'restaurant-name',
 'restaurant-semi-pricerange': 'restaurant-pricerange',
 'taxi-semi-arriveBy': 'taxi-arriveBy',
 'taxi-semi-departure': 'taxi-departure',
 'taxi-semi-destination': 'taxi-destination',
 'taxi-semi-leaveAt': 'taxi-leaveAt',
 'train-book-people': 'train-book_people',
 'train-semi-arriveBy': 'train-arriveBy',
 'train-semi-day': 'train-day',
 'train-semi-departure': 'train-departure',
 'train-semi-destination': 'train-destination',
 'train-semi-leaveAt': 'train-leaveAt'}

def load_data(version):
    with open(f'../data/MULTIWOZ{version}/train_dials.json', 'r') as f:
        train_data = json.load(f)

    with open(f'../data/MULTIWOZ{version}/dialogue_acts.json', 'r') as f:
        dialogue_dicts = json.load(f)

    return train_data, dialogue_dicts

def load_ontology(filename):
    onto_data_mod = {}
    with open(filename, 'r') as f:
        onto_data = json.load(f)

    ## processing to get it to a normalised slot format
    for key, val_list in onto_data.items():
        if(key in slot_maps):
            onto_data_mod[slot_maps[key]]  = val_list
    
    return onto_data_mod
def get_random_choice(slot, val,onto_data, chances = 10):
    
    for i in range(chances):
        new_val = random.choice(onto_data[slot])
        if(new_val == val):
            continue
        if(len(new_val.split()) == len(val.split())):
            return new_val, True
    return val, False

def augment_data_paraphrase(version, multi):
    train_data, dialogue_dicts = load_data(version)
    if(multi):
        outdir = f'../data/MULTIWOZ{version}-paraphrase-multi/'
    else:
        outdir = f'../data/MULTIWOZ{version}-paraphrase/'
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
                if multi:
                    out = augmentation.get_paraphrased_sentences(turn['text'], num_return_sequences=5)
                    out = sorted(out, key=lambda x: len(x.split()), reverse=True)
                else:
                    out = augmentation.get_paraphrased_sentences(turn['user'], num_return_sequences=1)
                turn['text'] = out[0]
                # turn['text'] = augmentation.get_paraphrased_sentences(turn['text'], num_return_sequences=1)[0] 
        train_data[ex_name + '_augment.json'] = dial_aug
        dialogue_dicts[ex_name + '_augment'] = dst_aug
        train_aug_count += 1

    print(len(train_data), len(dialogue_dicts))

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f, indent=4)
    
    
    return

def augment_data_translate(version):
    train_data, dialogue_dicts = load_data(version)

    outdir = f'../data/MULTIWOZ{version}-translate/'
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
        
        train_data[ex_name + '_augment.json'] = dial_aug
        dialogue_dicts[ex_name + '_augment'] = dst_aug
        train_aug_count += 1

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f, indent=4)

    return

def augment_data_crop_rotate(version, operation):

    LENGTH_CONSTRAINT = 0.6
    train_data, dialogue_dicts = load_data(version)

    outdir = f'../data/MULTIWOZ{version}-{operation}-multi/'
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
                translated_sent = augmentation.get_augmentation(orig_sent, operation)
                if(translated_sent !=  orig_sent):
                    translated_sent_len = len(translated_sent.split())
                    stats.append((orig_sent_len, translated_sent_len))
                    if(translated_sent_len > (LENGTH_CONSTRAINT*orig_sent_len)):
                        turn['text'] = translated_sent    
                        flag = True
        if(flag == True):
            train_data[ex_name + '_augment.json'] = dial_aug
            dialogue_dicts[ex_name + '_augment'] = dst_aug
            train_aug_count += 1

    print(len(train_data), len(dialogue_dicts))

    print(f'# training data: {train_count}')
    print(f'# Augmented data: {train_aug_count}')
    print(f'# training data after augmentation: {train_count + train_aug_count}')

    with open(f'{outdir}/train_dials.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(f'{outdir}/dialogue_acts.json', 'w') as f:
        json.dump(dialogue_dicts, f, indent=4)

    return

def augment_data_sequential(version, num_sequence=3):
    train_data, dialogue_dicts = load_data(version)
    outdir = f'../data/MULTIWOZ{version}-sequential/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_count = len(train_data)
    train_aug_count = 0

    for fn in tqdm(list(train_data.keys())):
        dial = train_data[fn]
        dial_aug = deepcopy(dial)

        ex_name = fn.split('.')[0] ## gives the example name without json tag
        dst = dialogue_dicts[ex_name]
        dst_aug = deepcopy(dst)
        log_turns = len(dial_aug['log'])
        num_turns = len(dst)

        for i in range(log_turns - 2*num_sequence + 1):
            dial_aug['log'][i]['text'] = ' '.join(dial_aug['log'][i+2*k]['text'] for k in range(num_sequence))


        train_data[ex_name + '_augment.json'] = dial_aug
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

## this function saves the output augmented examples into a pickle file
## the augmented examples are in the pre-processed format required for trippy and can be directly loaded before feature extraction
def get_entity_augment_func(train_data, dialogue_dicts, example, onto_data, guid_to_keys):
    new_example_list = []
    augmented_data = 0
    for idx, dial_id in enumerate(train_data):
        print(idx, dial_id)
        ex_name = dial_id.split('.')[0]
        turns = len(dialogue_dicts[ex_name])
        all_dial_example_ids = []
        example_key = 'train-'+dial_id+'-'+str(0)
        if example_key not in guid_to_keys:
            print("This key is not present : " , example_key)
            continue
        for turn_id in range(turns):
            example_key = 'train-'+dial_id+'-'+str(turn_id)
            if example_key not in guid_to_keys:
                continue
            example_id = guid_to_keys[example_key]
            all_dial_example_ids.append(example_id)
        print(example_id)
        
        final_val_dict = example[example_id].values
        final_val_dict_modified = deepcopy(final_val_dict)
        edited = False
        edited_slots = []
        for slot, val in final_val_dict.items():
            ## avoiding the slots that have numerical values
            if('book' in slot or 'people' in slot or 'stars' in slot):
                continue
            if (val!='none'):
                new_val, val_change_flag = get_random_choice(slot, val, onto_data)
                if(val_change_flag ==False):
                    continue
                ispresent_dial_flag = False
                for id_no in all_dial_example_ids:
                    if (val in example[id_no].text_a or val in example[id_no].text_b):
                        ispresent_dial_flag = True
                if(ispresent_dial_flag):
                    final_val_dict_modified[slot]=new_val
                    edited = True
                    edited_slots.append((slot, val,new_val) )
        
        ## when altleast one of it is edited
        if(len(edited_slots) == 0):
            continue

        for e_id in all_dial_example_ids:
            new_example = deepcopy(example[e_id])
            new_example.guid = 'train-'+ex_name+'_augment.json' + '-'+example[e_id].guid.split('-')[-1]
            for edit_slot, val, new_val  in edited_slots:
                new_example.text_a = [new_val if item == val else item for item in new_example.text_a]
                new_example.text_b = [new_val if item == val else item for item in new_example.text_b]
                new_example.history = [new_val if item == val else item for item in new_example.history]
                if (new_example.values[edit_slot] == val):
                    new_example.values[edit_slot] = new_val
                    
                if (new_example.inform_label[edit_slot] == val):
                    new_example.inform_label[edit_slot] = new_val
            new_example_list.append(new_example)
        augmented_data+=1
    
                    
    total_augmented_examples = example + new_example_list
    with open('./only_augmented_examples_2_1.pkl', 'wb') as f:
        pickle.dump(new_example_list, f)
    
    with open('./entity_replacement_2_1.pkl', 'wb') as f:
        pickle.dump(total_augmented_examples, f)

    print("Number of augmented examples", augmented_data)
    print(len(example), len(new_example_list), len(total_augmented_examples))

    return

def augment_data_entity_replacement(version):
    train_data, dialogue_dicts = load_data(version)
    onto_data = load_ontology('./ontology.json')

    print("Done loading data... \n Loading the examples now")
    ## take this from the saved examples in the trippy load_and_cache examples -- easier to modify after the pre-processing is done
    processed_examples_path = './example_2_1.pkl'
    with open(processed_examples_path, 'rb') as f_ex:
        example = pickle.load(f_ex)

    guid_to_keys = {}
    for id_no in range(len(example)):
        guid_to_keys[example[id_no].guid] = id_no


    get_entity_augment_func(train_data, dialogue_dicts, example, onto_data, guid_to_keys)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--version", type=str, help="MultiWOZ dataset version")
    args = parser.parse_args()

    if args.mode == 'paraphrase':
        augment_data_paraphrase(args.version, multi=False)
    elif args.mode == 'paraphrase_multi':
        augment_data_paraphrase(args.version, multi=True)
    elif args.mode == 'translate':
        augment_data_translate(args.version)
    elif args.mode == 'rotate':
        augment_data_crop_rotate(args.version, 'rotate')
    elif args.mode == 'crop':
        augment_data_crop_rotate(args.version, 'crop')
    elif args.mode == 'sequential':
        augment_data_sequential(args.version)
    elif args.mode == 'entity_replacement':
        augment_data_entity_replacement(args.version)

