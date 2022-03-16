from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
 
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
import argparse
from tqdm import tqdm
from pprint import pprint
import os

## loading the default config and tokenizer

def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained(args.load_ckpt) 

    model = BertForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    print("Loading the dataset ... ")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.data_path,
        block_size=128,
    )
    print("Done with loading data")


    training_args = TrainingArguments(
        output_dir=args.save_ckpt,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_gpu_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # prediction_loss_only=True
    )


    print ('Start a trainer...')
    # Start training
    trainer.train()
    
    # Save
    trainer.save_model(args.save_ckpt)
    print ('Finished training all...',args.save_ckpt)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", dest="data_path", action="store", required=True)
	parser.add_argument("--load_ckpt", dest="load_ckpt", action="store", required=False, default= 'bert-base-uncased')
	parser.add_argument("--save_ckpt", dest="save_ckpt", action="store", required=True)
	parser.add_argument("--num_epochs", dest="num_epochs", action="store", default=3, type=int)
	args = parser.parse_args()
	pprint(args)

	main(args)