## Introduction

TripPy is a new approach to dialogue state tracking (DST) which makes use of various copy mechanisms to fill slots with values. Our model has no need to maintain a list of candidate values. Instead, all values are extracted from the dialog context on-the-fly.
A slot is filled by one of three copy mechanisms:
1. Span prediction may extract values directly from the user input;
2. a value may be copied from a system inform memory that keeps track of the system’s inform operations;
3. a value may be copied over from a different slot that is already contained in the dialog state to resolve coreferences within and across domains.
Our approach combines the advantages of span-based slot filling methods with memory methods to avoid the use of value picklists altogether. We argue that our strategy simplifies the DST task while at the same time achieving state of the art performance on various popular evaluation sets including MultiWOZ 2.1.

## How to run experiments for this project

Use the script in `DO.example.simple`. You can to edit the following variables in the script to achieve the desired experiments:
-  `DATA_DIR` : Path to the dataset variation (including augmentations) 
- `TASK` : Dataset config
- `bert_model_path` :Path to the pre-trained bert model or `bert-base-uncased` for baseline
- `OUT_DIR`: Path to store the results


## Evaluate

- `metric_bert_dst.py` - Use this for Joint Goal Accuracy
- `metric_slot_scores.py` - Use this for slot F1 and slot accuracy
- `metric_domain.py` - Domain wise analysis on the performance

Usage for all these metrics files is as follows:

```
python metric_bert_dst <path to eval_pred in the outut directory>
```

## Datasets

Supported datasets are:
- sim-M (https://github.com/google-research-datasets/simulated-dialogue.git)
- sim-R (https://github.com/google-research-datasets/simulated-dialogue.git)
- WOZ 2.0 (see data/)
- MultiWOZ 2.0 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.1 (see data/, https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.2 (https://github.com/budzianowski/multiwoz.git)
- MultiWOZ 2.3 (https://github.com/lexmen318/MultiWOZ-coref.git)
- MultiWOZ 2.4 (https://github.com/smartyfh/MultiWOZ2.4.git)

With a sequence length of 180, you should expect the following average JGA:
- 56% for MultiWOZ 2.1
- 88% for sim-M
- 90% for sim-R
- 92% for WOZ 2.0

## Requirements

- torch (tested: 1.4.0)
- transformers (tested: 2.9.1)
- tensorboardX (tested: 2.0)

## Citation

This work is published as [TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://www.aclweb.org/anthology/2020.sigdial-1.4/)

If you use TripPy in your own work, please cite our work as follows:

```
@inproceedings{heck2020trippy,
    title = "{T}rip{P}y: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking",
    author = "Heck, Michael and van Niekerk, Carel and Lubis, Nurul and Geishauser, Christian and
              Lin, Hsien-Chin and Moresi, Marco and Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 21st Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "35--44",
}
```

This repository also contains the code of our paper [Out-of-Task Training for Dialog State Tracking Models"](https://www.aclweb.org/anthology/2020.coling-main.596).

If you use TripPy for MTL, please cite our work as follows:

```
@inproceedings{heck2020task,
    title = "Out-of-Task Training for Dialog State Tracking Models",
    author = "Heck, Michael and Geishauser, Christian and Lin, Hsien-chin and Lubis, Nurul and
              Moresi, Marco and van Niekerk, Carel and Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    pages = "6767--6774",
}
```
