from ast import dump
import torch
import random 
import numpy as np

filename = "cached_train_features_advanced_crop"
example_features = torch.load(filename)

guid_to_idx = {}

for id in range(len(example_features)):
    guid_to_idx[example_features[id].guid] = id

all_samples = list(set([key.split('-')[1] for key in guid_to_idx.keys()]))


augment_samples = [key for key in all_samples if '_augment' in key ]
real_samples = list(set(all_samples) - set(augment_samples))

print(len(all_samples), len(augment_samples), len(real_samples)) 
random.shuffle(augment_samples)
q1, q2, q3, q4 = np.array_split(augment_samples, 4)


real_samples_list, q1_list, q2_list, q3_list, q4_list = [], [], [], [], []
for id in range(len(example_features)):
    guid_key = example_features[id].guid.split('-')[1]
    if(guid_key in real_samples):
        real_samples_list.append(example_features[id])
    elif (guid_key in q1):
        q1_list.append(example_features[id])
    elif (guid_key in q2):
        q2_list.append(example_features[id])
    elif (guid_key in q3):
        q3_list.append(example_features[id])
    elif (guid_key in q4):
        q4_list.append(example_features[id])
print(len(real_samples_list), len(q1_list), len(q2_list), len(q3_list), len(q4_list))
write_file = './cache_data/'+filename
dump_data = real_samples_list
dump_data = dump_data + q1_list
print("Saving 25 percent ... ", len(dump_data))
torch.save(dump_data, write_file + '_25_1')

dump_data = dump_data + q2_list
print("Saving 50 percent ... ", len(dump_data))
torch.save(dump_data, write_file + '_50_1')

dump_data = dump_data + q3_list
print("Saving 75 percent ... ", len(dump_data))
torch.save(dump_data, write_file + '_75_1')
