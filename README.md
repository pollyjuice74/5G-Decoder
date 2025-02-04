# 5G LDPC Linear Transformer for Channel Decoding

Implementation of the linear transformer-based decoder experiments from
[*5G LDPC Linear Transformer for Channel Decoding*](https://arxiv.org/abs/2501.14102) using the
[Sionna link-level simulator](https://nvlabs.github.io/sionna/).

## Abstract
This work introduces a novel, fully differentiable linear-time complexity transformer decoder and a transformer decoder to correct 5G New Radio (NR) LDPC codes. We propose a scalable approach to decode linear block codes with `O(n)` complexity rather than `O(n^2)` for regular transformers. The architectures' performances are compared to Belief Propagation (BP), the production-level decoding algorithm used for 5G New Radio (NR) LDPC codes. We achieve bit error rate performance that matches a regular Transformer decoder and surpases one iteration BP, also achieving competitive time performance against BP, even for larger block codes. We utilize Sionna, Nvidia's 5G \& 6G physical layer research software, for reproducible results.

## Structure of this repository

- `LTD_model_5G_LDPC.ipynb` running Decoder on 5G LDPC codes.

In `src/` folder: 
  
- `decoder.py` linear and regular transformer decoder model.
- `decoder5G.py` decoder model for 5G LDPC codes.
- `e2e_model.py` end-to-end channel between Encoder and Decoder.
- `args.py` class holding arguments for the models and training.
- `utils.py` save, load, train/test functions for transformer models.
- `utils5G.py` 5G pcm pruning function.
- `time_comparison.py` comparison and plotting functions to evaluate speed of models.


## References

[A] [M. Hernandez, F. Pinero, "5G LDPC Linear Transformer for Channel Decoding", 2025]()

[B] [S. Cammerer, J. Hoydis, F. Aït Aoudia, and A. Keller, "Graph Neural Networks for Channel Decoding", 2022](https://arxiv.org/pdf/2207.14742.pdf)

[C] [Yoni Choukroun, Lior Wolf, "Error Correction Code Transformer", 2022](https://arxiv.org/abs/2203.14966)


## License

© 2025 Mario Hernandez Torres.
This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).  
Distributed on an "AS IS" basis, without warranties or conditions of any kind. 
