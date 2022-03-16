import os
import torch
import random
import argparse
from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Union

from transformers import (
    PreTrainedTokenizer,
    BertForMaskedLM,
    BertTokenizer,
    Trainer,
    BertConfig,
    TrainingArguments,
)

class MethodDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int=256):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = []
        for line in tqdm(lines):
            input_encodings = tokenizer.encode_plus(line, pad_to_max_length=True, max_length=block_size, truncation=True)
            encodings = {'input_ids': torch.tensor(input_encodings['input_ids'] , dtype=torch.long)}
            self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]

# Translated from: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
@dataclass
class DataCollatorForSpanLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], dict):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability = 0.15, min_span_length = 1, max_span_length = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        device = inputs.device
        inpts = inputs.clone()
        span_lengths = torch.randint(low = min_span_length, high = max_span_length + 1, size = (inpts.shape[0],), device = device)
        periods = torch.round(span_lengths / mlm_probability)
        offsets = torch.tensor([random.randint(0, period.item()) for period in periods], device = device)
        masks = torch.stack([(torch.arange(start = 0, end = inpts.shape[1]) + offset) % period.long() < span for offset, period, span in zip(offsets, periods, span_lengths)])

        if self.tokenizer._pad_token is not None:
            padding_mask = inpts.eq(self.tokenizer.pad_token_id)
            masks.masked_fill_(padding_mask, value = False)
        # num_masks = torch.floor_divide(masks.sum(axis = 1), span_lengths)
        num_masks = torch.div(masks.sum(axis = 1), span_lengths)
        # new_inpts = []
        # lbls = []
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # for inpt, mask in zip(inpts, masks):
        #     new_inpts.append(
        #         self._noise_span_to_unique_sentinel(inpt, mask, 100, self.tokenizer.convert_tokens_to_ids(['mask'])[0])
        #     )
        #     lbls.append(
        #         self._noise_span_to_unique_sentinel(inpt, ~mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
        #     )

        labels = inputs.clone()
        inputs[masks] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels[~masks] = -100  # We only compute loss on masked tokens

        # import pdb
        # pdb.set_trace()        
        # new_inpts = pad_sequence(new_inpts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # lbls = pad_sequence(lbls, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return inputs, labels

def main(args):
    pre_trained_bert_model =  "bert-base-uncased"
    print('Loading the default tokenizer')
    tokenizer = BertTokenizer.from_pretrained(pre_trained_bert_model)
    # tokenizer.add_special_tokens({'mask_token': '<mask>'})

    print('Loading data')
    dataset = MethodDataset(tokenizer=tokenizer, file_path=args.data_path, block_size=128)
    data_collator = DataCollatorForSpanLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    print('Loading pretrained model')
    
    config = BertConfig.from_pretrained(pre_trained_bert_model) 
    model = BertForMaskedLM.from_pretrained(args.load_ckpt)

    print('Starting model training')
    training_args = TrainingArguments(
        output_dir=args.save_ckpt,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # prediction_loss_only=True
    )

    trainer.train()

    print('Saving model')
    trainer.save_model(args.save_ckpt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", action="store", required=True, default='./temp.txt')
    parser.add_argument("--load_ckpt", dest="load_ckpt", action="store", required=False, default= 'bert-base-uncased')
    parser.add_argument("--save_ckpt", dest="save_ckpt", action="store", required=True, default='./tmp')
    parser.add_argument("--num_epochs", dest="num_epochs", action="store", default=3, type=int)
    args = parser.parse_args()
    pprint(args)

    main(args)