## Pre-training Strategies for Different Transformer Architectures

### Pre-training Techniques

- `./bert_pretraining.py`: Pretraining BERT models with Masked Language Modelling on custom dataset
- `./span_bert_pretraining.py`: Pretraining BERT models with Span Prediction on custom dataset
- `./t5_pretraining.py`: Pretraning T5 with Span Curroption task on custom dataset

Example usage for all the pre-training files:

```
python t5_pretraining.py --data_path {custom_data_path} --load_ckpt {base_model} --save_ckpt {save_ckpt} --num_epochs {num_epochs}
```

where options for various arguments are
- `{custom_data_path}` Path to the custom dataset on which we want to pre-train
- `{base_model}` Base transformer model in huggingface like `t5-small`, `bert-base-uncased` or a checkpoint of the already trained model
- `{save_ckpt}` Path to save the model after pre-training
- `{num_epochs}` Number of epochs to pre-train it on.


### Analysis
- `./vocabulary_overlap.py`: Visualise the vocabulary overlap between generic book corpus and task-related dialogue corpuses
