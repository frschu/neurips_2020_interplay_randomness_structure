# "The interplay between randomness and structure during learning in RNNs"

This repository accompanies the NeurIPS 2020 conference paper [The interplay between randomness and structure during learning in RNNs](https://arxiv.org/abs/2006.11036). 

The repository contains code to reproduce the results presented in the paper: training the RNN models and producing the figures with data generated from training. 

To train models, run the generate_\*.ipynb notebooks. All data is saved in the directory 'data/' (this can be changed globally in 'data_dir.py').

To reproduce the figures, run the figures_\*.ipynb notebooks. The training data files may need to be adjusted according to the files available (training parameters are partially kept in the file names). All figures are save to the directory 'figures/' (this can be changed globally in 'fig_specs.py').

The file names correspond to the following parts of the paper / figures:
* 'linear': linear RNN trained on the simplified input-driven fixed point task
* 'nonlinear': nonlinear RNN trained on the three neuroscience tasks ('flipflop', 'mante', 'romo').
* 'sentiment_analysis': LSTM model trained on the sentiment analysis task
* 'supp_cosine': linear RNN trained to generate a cosine function (supplementary)
* 'supp_norm_scale_with_N': nonlinear RNN trained on the three neuroscience tasks, iterating over network size N

## Requirements

* python 3.8
* numpy 1.18.5
* pytorch 1.6.0
* matplotlib 3.2.2

For the NLP task, we additionally need

* torchtext 0.7.0
* spacy 2.3.2 (tokenizer)
