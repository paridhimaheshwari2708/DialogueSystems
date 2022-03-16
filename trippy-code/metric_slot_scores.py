# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import sys
import numpy as np
import re


def load_dataset_config(dataset_config):
    with open(dataset_config, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    return raw_config['class_types'], raw_config['slots'], raw_config['label_maps']


def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value_label, inform_label, label_maps):
    value = inform_label
    if value_label == inform_label:
        value = value_label
    elif is_in_list(inform_label, value_label):
        value = value_label
    elif is_in_list(value_label, inform_label):
        value = value_label
    elif inform_label in label_maps:
        for inform_label_variant in label_maps[inform_label]:
            if value_label == inform_label_variant:
                value = value_label
                break
            elif is_in_list(inform_label_variant, value_label):
                value = value_label
                break
            elif is_in_list(value_label, inform_label_variant):
                value = value_label
                break
    elif value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            if value_label_variant == inform_label:
                value = value_label
                break
            elif is_in_list(inform_label, value_label_variant):
                value = value_label
                break
            elif is_in_list(value_label_variant, inform_label):
                value = value_label
                break
    return value


def get_joint_slot_correctness(preds, class_types, label_maps,
                               key_class_label_id='class_label_id',
                               key_class_prediction='class_prediction',
                               key_start_pos='start_pos',
                               key_start_prediction='start_prediction',
                               key_end_pos='end_pos',
                               key_end_prediction='end_prediction',
                               key_refer_id='refer_id',
                               key_refer_prediction='refer_prediction',
                               key_slot_groundtruth='slot_groundtruth',
                               key_slot_prediction='slot_prediction'):

    class_correctness = [[] for cl in range(len(class_types) + 1)]
    confusion_matrix = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
    pos_correctness = []
    refer_correctness = []
    val_correctness = []
    total_correctness = []
    c_tp = {ct: 0 for ct in range(len(class_types))}
    c_tn = {ct: 0 for ct in range(len(class_types))}
    c_fp = {ct: 0 for ct in range(len(class_types))}
    c_fn = {ct: 0 for ct in range(len(class_types))}

    tp,tn,fp,fn =0,0,0,0

    for pred in preds:
        guid = pred['guid']  # List: set_type, dialogue_idx, turn_idx
        turn_gt_class = pred[key_class_label_id]
        turn_pd_class = pred[key_class_prediction]
        gt_start_pos = pred[key_start_pos]
        pd_start_pos = pred[key_start_prediction]
        gt_end_pos = pred[key_end_pos]
        pd_end_pos = pred[key_end_prediction]
        gt_refer = pred[key_refer_id]
        pd_refer = pred[key_refer_prediction]
        gt_slot = pred[key_slot_groundtruth]
        pd_slot = pred[key_slot_prediction]

        gt_slot = tokenize(gt_slot)
        pd_slot = tokenize(pd_slot)

        # Make sure the true turn labels are contained in the prediction json file!
        joint_gt_slot = gt_slot
    
        if guid[-1] == '0': # First turn, reset the slots
            joint_pd_slot = 'none'

        # Check the joint slot correctness.
        # If the value label is not none, then we need to have a value prediction.
        # Even if the class_type is 'none', there can still be a value label,
        # it might just not be pointable in the current turn. It might however
        # be referrable and thus predicted correctly.
        if(pd_slot == gt_slot):
            if(pd_slot == 'none'):
                tn+=1
            else:
                tp +=1
            # if(pd_slot != 'none'):
            #     tp +=1
        else:
            if(gt_slot == 'none'):
                fp+=1
            else:
                fn+=1

    return np.asarray(total_correctness), np.asarray(val_correctness), np.asarray(class_correctness), np.asarray(pos_correctness), np.asarray(refer_correctness), np.asarray(confusion_matrix), c_tp, c_tn, c_fp, c_fn, tp,tn,fp,fn


if __name__ == "__main__":
    acc_list = []
    acc_list_v = []
    key_class_label_id = 'class_label_id_%s'
    key_class_prediction = 'class_prediction_%s'
    key_start_pos = 'start_pos_%s'
    key_start_prediction = 'start_prediction_%s'
    key_end_pos = 'end_pos_%s'
    key_end_prediction = 'end_prediction_%s'
    key_refer_id = 'refer_id_%s'
    key_refer_prediction = 'refer_prediction_%s'
    key_slot_groundtruth = 'slot_groundtruth_%s'
    key_slot_prediction = 'slot_prediction_%s'

    # dataset = sys.argv[1].lower()
    # dataset_config = sys.argv[2].lower()

    ### used as python metric_domain.py <pred file name obained after testing the model>

    dataset = "multiwoz20"
    dataset_config = "dataset_config/multiwoz21.json"

    if dataset not in ['woz2', 'sim-m', 'sim-r', 'multiwoz21', 'multiwoz20']:
        raise ValueError("Task not found: %s" % (dataset))

    class_types, slots, label_maps = load_dataset_config(dataset_config)

    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp

    print(sys.argv[1])

    for file_path in sorted(glob.glob(sys.argv[1])):
        print(file_path)
        with open(file_path) as f:
            preds = json.load(f)
        

        goal_correctness = 1.0
        cls_acc = [[] for cl in range(len(class_types))]
        cls_conf = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
        c_tp = {ct: 0 for ct in range(len(class_types))}
        c_tn = {ct: 0 for ct in range(len(class_types))}
        c_fp = {ct: 0 for ct in range(len(class_types))}
        c_fn = {ct: 0 for ct in range(len(class_types))}
        total_tp,total_tn,total_fp,total_fn = 0,0,0,0
        for slot in slots:
            tot_cor, joint_val_cor, cls_cor, pos_cor, ref_cor, conf_mat, ctp, ctn, cfp, cfn, tp,tn,fp,fn = get_joint_slot_correctness(preds, class_types, label_maps,
                                                             key_class_label_id=(key_class_label_id % slot),
                                                             key_class_prediction=(key_class_prediction % slot),
                                                             key_start_pos=(key_start_pos % slot),
                                                             key_start_prediction=(key_start_prediction % slot),
                                                             key_end_pos=(key_end_pos % slot),
                                                             key_end_prediction=(key_end_prediction % slot),
                                                             key_refer_id=(key_refer_id % slot),
                                                             key_refer_prediction=(key_refer_prediction % slot),
                                                             key_slot_groundtruth=(key_slot_groundtruth % slot),
                                                             key_slot_prediction=(key_slot_prediction % slot)
                                                             )
            # print('%s: joint slot acc: %g, joint value acc: %g, turn class acc: %g, turn position acc: %g, turn referral acc: %g' %
                #   (slot, np.mean(tot_cor), np.mean(joint_val_cor), np.mean(cls_cor[-1]), np.mean(pos_cor), np.mean(ref_cor)))
            goal_correctness *= tot_cor

            total_tp += tp 
            total_tn += tn 
            total_fp += fp 
            total_fn += fn 

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        slot_acc = (total_tp +total_tn) / (total_tp + total_fp +total_fn + total_tn+ 1e-10)

        acc = np.mean(goal_correctness)
        acc_list.append((fp, acc,slot_acc, f1, precision, recall ))
        

    acc_list_s = sorted(acc_list, key=lambda tup: tup[1], reverse=True)
    for (fp, acc, slot_acc, f1, precision, recall) in acc_list_s:
        # print('Joint goal acc: %g, %s' % (acc, fp))
        print('Slot Accuracy : %s, Precision: %s, Recall: %s, F1 Score: %s' % (slot_acc, precision, recall, f1 ))
