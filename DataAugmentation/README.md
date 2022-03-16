## Data Augmentation Techniques for MultiWOZ dataset

### Data Augmentation Strategies:
- Entity Replacement
- Crop and Rotate with dependency parse trees
- Paraphrasing with PEGASUS
- Translate 
- Sequential addition

### File structure:
- `./augment_mintl.py`: Code for different augmentations in the required format for MinTL baseline
- `./augment_trippy.py`: Code for different augmentations in the required format for TripPy baselike

Usage:

```
python augment_mintl.py --mode {augmentation_strategy} --version {version}
```

where options for various arguments are
- `{augmentation_strategy}` Can be `paraphrase_multi`, `translate`, `entity_replacement`, `crop`, `rotate`, `sequential`
- `{version}` MultiWOZ data version - either `2.0` or `2.1`
