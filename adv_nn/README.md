# Adversarial Neural Network Implementation

Contains implementation a **tensorflow** implementation for scalability of a `Discriminator` and a `Generator` where the discriminator decodes a recieved codeword `r`
and generator adds noise `z_G` to a codeword `c` such that it will increase the discriminator's `BER` performance 
on a given `EbN0` signal to noise ratio. 

The `Discriminator` is implemented using low-rank projections and splitting numerical methods to decrease the time complexity 
of the transformer and diffusion respectively such that it will have competetive time complity with existing GNN/BP 5G 
decoding approaches of `O( Iters * |edges(H)| )`. 

The `Generator` is implemented similarly.

---

## Overview

- `LinearTranDiff.ipynb` running Discriminator on datasets and making sure layers work properly.

- `models.py` Discriminator and Generator models.
- `channel.py` end-to-end channel between Discriminator and Generator.
  
- `decoder.py` transformer class.
- `attention.py` low-rank projection implementation of attention with `O( (n+m)d * |edges(H_mask)| )`.

- `dataset.py` dataset with dataset types: zero_cw, ones_m, flip_cw, random_bits.
- `args.py` class holding arguments for the models and datasets.

## Resources
