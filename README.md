# 5G LDPC Decoder

## Overview

- `LinearTranDiff.ipynb` running Decoder/Discriminator on datasets and making sure layers work properly.

In `adv_nn/` folder: 
- `models.py` Discriminator and Generator models.
- `model_functs.py` train/test functions for models.
- `channel.py` end-to-end channel between Discriminator and Generator.
  
- `decoder.py` transformer class.
- `attention.py` low-rank projection implementation of attention with `O( (n+m)d * |edges(H_mask)| )`.

- `dataset.py` dataset with dataset types: zero_cw, ones_m, flip_cw, random_bits.
- `args.py` class holding arguments for the models and datasets.

