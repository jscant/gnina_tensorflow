# gnina_tensorflow: A TensorFlow implementation of models based on the gnina framework

Gnina is a method of featurisation of 3D protein-ligand complexes [[1]](#1) for input into convolutional neural networks. This repo is a collection of machine learning algorithms built on top of gnina. It can be used as is, but for testing to work properly a local installation is required:

```
cd gnina_tensorflow
pip install -e .
python3 -m pytest
```

## Models included

There are two virtual screening architectures: the original implementation [[1]](#1), and DenseFS [[2]](#2). There is a random forest, is trained for the same task but uses the final layer of gnina CNNs before classification as input (mainly used as a diagnostic for a paper which is in the works). There is also an autoencoder, which aims to reduce the dimensionality of the original gnina input.

## References
<a id="1">[1]</a> 
M Ragoza, J Hochuli, E Idrobo, J Sunseri, DR Koes. (2017). 
Protein–Ligand Scoring with Convolutional Neural Networks
[J. Chem. Inf. Model. 57, 4, 942–957](http://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740)

<a id="2">[2]</a> 
Imrie, F.; Bradley, A. R.; van der Schaar, M.; Deane, C. M. (2018). 
Protein Family-Specific Models Using Deep Neural Networks and Transfer Learning Improve Virtual Screening and Highlight the Need for More Data
[J. Chem. Inf. Model. 58, 2319−2330.](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00350)
