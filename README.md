# gnina_tensorflow: A TensorFlow implementation of models based on the gnina framework

Gnina is a method of featurisation of 3D protein-ligand complexes [[1]](#1) for input into convolutional neural networks. This repo is a collection of machine learning algorithms built on top of gnina. It can be used as is, but for testing to work properly a local installation is required:

```
cd gnina_tensorflow
pip install -e .
python3 -m pytest
```

## Requirements

The following are required for all or part of gnina_tensorflow:
```
scikit-learn
joblib
tensorflow>=2.4.0: pip install tf-nightly
```

Please note: 3D convolutions on the data format provided by gnina do not work when running TensorFlow on a CPU. Hopefully this will change in a future release of TensorFlow.

**Previous work using gnina by this author can be found in Ref. [[3]](#3)**

## Functionality included

There are two virtual screening architectures: the original implementation [[1]](#1), and DenseFS [[2]](#2). There is a random forest, is trained for the same task but uses low dimensional encodings of gnina grids as input (mainly used as a diagnostic for a paper which is in the works). These encodings are generated by an autoencoder, which aims to reduce the dimensionality of the original gnina input.

## Virtual Screening

Virtual screening is tasked with discriminating between active compounds and decoy compounds - that is, molecules that bind vs do not bind to a given protein target. An example of how to train a model on the small training set provided is shown below:

```
cd gnina_tensorflow
python3 classifier/gnina_tensorflow.py --data_root data --train data/small_chembl_test.types --batch_size 16 --iterations 100 --save_path classifier_example --densefs --inference_on_training_set
```

## Autoencoder

An autoencoder aims to learn a meaningful low-dimensional representation of a high-dimensional input. In this instance, gnina inputs which are similar in some sense (perhaps the proteins are the same and the ligands only differ by the location of a functional group on an aromatic ring) should be converted to encodings (vectors) which are a low distance apart; the inverse should hold true for dissimilar inputs.

Dimensionality reduction is an important part of my ongoing work with gnina; an example is show below:

```
cd gnina_tensorflow
python3 autoencoder/gnina_autoencoder.py --data_root data --train data/small_chembl_test.types --batch_size 1 --iterations 1000 --save_path autoencoder_example --encoding_size 200 --optimiser adamax --learning_rate 0.001 --final_activation sigmoid --dimension 18.0 --resolution 1.0 --save_encodings
```

## Random Forest

Random forests are used to test the quality of the encodings generated by the autoencoder. The autoencoder is first trained on, for example, the DUD-E dataset [[4]](#4); the resulting encodings are used to train a random forest to discriminate between actives and decoys. The same trained autoencoder is then used to generate encodings for a test set - like the *ChEMBL validation set* from Ref. [[3]](#3) - which are used to validate the model. If good performance is achieved on the test set, we can deduce that the encodings are good representations of the original inputs:

```
cd gnina_tensorflow
python3 classifier/random_forest.py rf_example --train_dir data/small_chembl_test_encodings
```

## References
<a id="1">[1]</a> 
M Ragoza, J Hochuli, E Idrobo, J Sunseri, DR Koes. (2017). 
Protein–Ligand Scoring with Convolutional Neural Networks
[J. Chem. Inf. Model. 57, 4, 942–957](http://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740)

<a id="2">[2]</a> 
Imrie, F.; Bradley, A. R.; van der Schaar, M.; Deane, C. M. (2018). 
Protein Family-Specific Models Using Deep Neural Networks and Transfer Learning Improve Virtual Screening and Highlight the Need for More Data
[J. Chem. Inf. Model. 58, 2319−2330.](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00350)

<a id="3">[3]</a>
Scantlebury, J.; Brown, N.; von Delft, F.; Deane, C. M. (2020).
Data Set Augmentation Allows Deep Learning-Based Virtual Screening to Better Generalize to Unseen Target Classes and Highlight Important Binding Interactions
[J. Chem. Inf. Model. XX, XXXX-XXXX.](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263)

<a id="4">[4]</a>
Mysinger, M. M.; Carchia, M.; Irwin, J. J.; Shoichet, B. K. (2012).
Directory of Useful Decoys, Enhanced (DUD-E): Better Ligands and Decoys for Better Benchmarking
[J. Med. Chem. 2012, 55, 14, 6582–6594.](https://pubs.acs.org/doi/10.1021/jm300687e)
