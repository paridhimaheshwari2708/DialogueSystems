import os
import re
import csv
import time
import json
import spacy
import torch
import random
import logging
import numpy as np
from copy import deepcopy
from itertools import chain
from collections import Counter, OrderedDict

from damd_multiwoz import ontology
from damd_multiwoz.db_ops import MultiWozDB
from damd_multiwoz.config import global_config as cfg

class Vocab(object):
    def __init__(self, model, tokenizer):
        self.special_tokens = ["pricerange", "<pad>", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "<None>", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","[db_state0+bookfail]", "[db_state1+bookfail]","[db_state2+bookfail]","[db_state3+bookfail]","[db_state4+bookfail]", "[db_state0+booksuccess]","[db_state1+booksuccess]","[db_state2+booksuccess]","[db_state3+booksuccess]","[db_state4+booksuccess]"]
        self.attr_special_tokens = {'pad_token': '<pad>',
                         'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "<None>", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","[db_state0+bookfail]", "[db_state1+bookfail]","[db_state2+bookfail]","[db_state3+bookfail]","[db_state4+bookfail]", "[db_state0+booksuccess]","[db_state1+booksuccess]","[db_state2+booksuccess]","[db_state3+booksuccess]","[db_state4+booksuccess]"]}
        self.tokenizer = tokenizer
        self.vocab_size = self.add_special_tokens_(model, tokenizer)


    def add_special_tokens_(self, model, tokenizer):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        #orig_num_tokens = model.config.vocab_size  #some of experiments use this...
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(self.attr_special_tokens) # doesn't add if they are already there
        
        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
            model.tie_decoder()
        # print(orig_num_tokens)
        # print(num_added_tokens)

        return orig_num_tokens + num_added_tokens

    def encode(self, word):
        """ customize for damd script """
        return self.tokenizer.encode(word)[0]

    def sentence_encode(self, word_list):
        """ customize for damd script """
        return self.tokenizer.encode(" ".join(word_list))

    def decode(self, idx):
        """ customize for damd script """
        return self.tokenizer.decode(idx)

    def sentence_decode(self, index_list, eos=None):
        """ customize for damd script """
        l = self.tokenizer.decode(index_list)
        l = l.split()
        if not eos or eos not in l:
            text = ' '.join(l)
        else:
            idx = l.index(eos)
            text = ' '.join(l[:idx])
        return puntuation_handler(text)

# T5 cannot seperate the puntuation for some reason
def puntuation_handler(text):
    text = text.replace("'s", " 's")
    text = text.replace(".", " .")
    text = text.replace("!", " !")
    text = text.replace(",", " ,")
    text = text.replace("?", " ?")
    text = text.replace(":", " :")
    return text

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i:i[0]))


    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            #print(f"batch_size{cfg.batch_size}")
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder < 1/10 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch)%len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch)%len(cfg.cuda_device))]
        if len(batch) > 0.1 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs


    def get_batches(self, set_name):
        global dia_count
        log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            if set_name != 'test' and k==1 or k>=17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n"%(
                    k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            all_batches += batches
        log_str += 'total batch num: %d\n'%len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)


    def save_result(self, write_mode, results, field, write_title=False):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None

    def save_result_report(self, results):
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv'%cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s'%str(cfg.beam_width)
            if cfg.beam_diverse_param>0:
                setting += ', penalty=%s'%str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s'%str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s'%str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn':cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
                  'decode': cfg.aspn_decode_mode, 'param':setting, 'nbest': cfg.nbest, 'selection_sheme': cfg.act_selection_scheme,
                  'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'], 'act_f1': results[0]['act_f1'],
                  'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])
