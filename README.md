# 5G LDPC Linear Transformer for Channel Decoding
Implementation of the linear transformer-based decoder experiments from
[*5G LDPC Linear Transformer for Channel Decoding*](link) using the
[Sionna link-level simulator](https://nvlabs.github.io/sionna/).

## Abstract
This work introduces a novel, fully differentiable linear-time complexity transformer decoder and a transformer decoder to correct 5G New Radio (NR) LDPC codes. We propose a scalable approach to decode linear block codes with `O(n)` complexity rather than `O(n^2)` for regular transformers. The architectures' performances are compared to Belief Propagation (BP), the production-level decoding algorithm used for 5G New Radio (NR) LDPC codes. We achieve bit error rate performance that matches a regular Transformer decoder and surpases one iteration BP, also achieving competitive time performance against BP, even for larger block codes. We utilize Sionna, Nvidia's 5G \& 6G physical layer research software, for reproducible results.

## Structure of this repository

- `LTD_model_5GLDPC.ipynb` running Decoder on 5G LDPC codes.

In `src/` folder: 
- `utils.py` train/test functions for transformer models.
- `utils5G.py` train/test functions for transformer models.
- `time_comparison.py` comparison and plotting functions to evaluate speed of models.
  
- `e2e_model.py` end-to-end channel between Encoder and Decoder.
- `decoder5G.py` decoder model for 5G LDPC codes.
- `decoder.py` linear and regular transformer decoder model.

- `args.py` class holding arguments for the models and training.
