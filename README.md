# 5G LDPC Decoder

**This research project is still in progress.**

## Adversarial Neural Network Implementation

Contains a **tensorflow** implementation for scalability of a `Decoder/Discriminator` and a `Generator` where the discriminator decodes a recieved codeword `r`
and generator adds noise `z_G` to a codeword `c` such that it will increase the discriminator's `BER` performance 
on a given `EbN0` signal to noise ratio. 

The `Decoder/Discriminator` is implemented using low-rank projections and splitting numerical methods to decrease the time complexity 
of the transformer and diffusion respectively such that it will have competetive time complity with existing GNN/BP 5G 
decoding approaches of `O( Iters * |edges(H)| )`. 

The `Generator` is implemented similarly.

The draft for the paper can be found [here](Linear_Transformer_Diffusion_Model.pdf) or the `Linear_Transformer_Diffusion_Model.pdf` file above.

## Results
It got 0 Bit Error Rate (decoded all errors) for EbNos between 5-13dB (5 is more noise like in a city with many wifi and cellular signals interfering, 10-20 is similar to a open space with less signal overlap) from a custom dataset that adds noise based on a normal distribution and a rayleigh distribution that simulates real world multipath noise (noise overlap from multiple sources). It takes a recieved vector `y = f(x) + z`, where `x` is a all zeros vector that is the original binary bits sent, `f` is a phase shift keying mapping binary bits to a real number signal to send through the channel, `z` is a real number noise vector form the normal and rayleigh distributions. 

**Speed Imporovement:** Time it decodes on a Google Colab TPU was 160 vectors of size (100,) with a parity check matrix of size (100,50) at 0.07 seconds. 

These results can be found on the `LTD_model_reg_LDPC.ipynb` notebook.

The relative time it decodes on a Google Colab TPU was 100 vectors of size (121,) with a parity check matrix of size (121,44) at 60 seconds. 

These results can be found on the `notebooks/LinearTranDiff.ipynb` notebook.


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


## TODO
- Decode LDPC codes on Sionna's functions for constructing parity check matrixes.
- Decode LDPC codes on 5G protocol.
- Adversarial Training
- Compare performance results to current state of the art 5G decoders.


## Resources

--- 

#### Most important source code on how LDPC 5G PCMs are constructed

[LDPC5GEncoder](https://nvlabs.github.io/sionna/api/fec.ldpc.html#sionna.fec.ldpc.encoding.LDPC5GEncoder) Docs

https://nvlabs.github.io/sionna/_modules/sionna/fec/ldpc/encoding.html#LDPC5GEncoder Source Code

--- 
- Transformer optimal time complexity (Linformer)

https://arxiv.org/pdf/2006.04768

- Transformer most accurate

https://arxiv.org/pdf/1905.07799

- Diffusion optimal time complexity (spliting numerical methods)

https://arxiv.org/pdf/2301.11558

- Diffusion memory efficient

https://arxiv.org/pdf/2010.02502

- Generative Adversarial Network variants

https://github.com/hwalsuklee/tensorflow-generative-model-collections/tree/master 

- General, powerful, scalable graph transformer layer

https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GPSConv.html#torch_geometric.nn.conv.GPSConv

- Graph Diffusion

https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.GDC.html?highlight=diffusion

- GNN BP Decoder model

[https://github.com/NVlabs/gnn-decoder/blob/master/gnn.py#L233](https://github.com/NVlabs/gnn-decoder/blob/master/gnn.py#L67)

- Error Correcting Code Transformer model

[https://github.com/yoniLc/ECCT/blob/main/Model.py](https://github.com/yoniLc/ECCT/blob/main/Model.py#L106)

- Creating Message Passing GNN

https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html


## Extra links

- Performers

https://arxiv.org/abs/2009.14794
