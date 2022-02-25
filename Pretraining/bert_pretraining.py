from transformers import (
	BertTokenizer,
	BertConfig,
	BertForMaskedLM,
	DataCollatorForLanguageModeling,
	LineByLineTextDataset,
	Trainer,
	TrainingArguments,
)

# DATA_PATH = 'all_data.txt'
# SAVE_CKPT_PATH = './all_bert/'
# DATA_PATH = 'multiwoz_data.txt'
# SAVE_CKPT_PATH = './multiwoz_bert/'
DATA_PATH = 'multiwoz_seq_data.txt'
SAVE_CKPT_PATH = './multiwoz_seq_bert/'

NUM_EPOCHS = 3

print('Loading the default config and tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')

print('Loading data')
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=DATA_PATH, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

print('Loading pretrained model')
model = BertForMaskedLM(config=config)

print('Starting model training')
training_args = TrainingArguments(
	output_dir=SAVE_CKPT_PATH,
	overwrite_output_dir=True,
	num_train_epochs=NUM_EPOCHS,
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
)

trainer.train()

print('Saving model')
trainer.save_model(SAVE_CKPT_PATH)
