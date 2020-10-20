# "The interplay between randomness and structure during learning in RNNs"

This repository accompanies the NeurIPS 2020 conference paper [The interplay between randomness and structure during learning in RNNs](https://arxiv.org/abs/2006.11036). 

The repository contains code to reproduce the results presented in the paper: training the RNN models and producing the figures with data generated from training. 

To train models, run the generate_\*.ipynb notebooks. All data is saved in the directory './data/' (this can be changed globally in 'data_dir.py').

To reproduce the figures, run the figures_\*.ipynb notebooks. The training data files may need to be adjusted according to the files available (training parameters are partially kept in the file names).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...
