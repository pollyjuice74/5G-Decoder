# 5G LDPC Linear Transformer for Channel Decoding

## Abstract
This work introduces a novel, fully differentiable linear-time complexity transformer decoder and a transformer decoder to correct 5G New Radio (NR) LDPC codes. We propose a scalable approach to decode linear block codes with `O(n)` complexity rather than `O(n^2)` for regular transformers. The architectures' performances are compared to Belief Propagation (BP), the production-level decoding algorithm used for 5G New Radio (NR) LDPC codes. We achieve bit error rate performance that matches a regular Transformer decoder and surpases one iteration BP, also achieving competitive time performance against BP, even for larger block codes. We utilize Sionna, Nvidia's 5G \& 6G physical layer research software, for reproducible results.

## Overview

- `LTD_model_5GLDPC.ipynb` running Decoder on 5G LDPC codes.

In `adv_nn/` folder: 
- `models.py` Discriminator and Generator models.
- `model_functs.py` train/test functions for models.
- `channel.py` end-to-end channel between Discriminator and Generator.
  
- `decoder.py` transformer class.
- `attention.py` low-rank projection implementation of attention with `O( (n+m)d * |edges(H_mask)| )`.

- `dataset.py` dataset with dataset types: zero_cw, ones_m, flip_cw, random_bits.
- `args.py` class holding arguments for the models and datasets.

