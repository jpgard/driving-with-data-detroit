# Detroit Vehicles 

Note: the data used for this analysis is not publicly available and is governed by a data non-disclosure agreement with the City of Detroit Operations and Infrastructure Group. This repository contains the complete source files used to generate the graphics and analysis in the paper:

```
Driving with Data: Modeling and Forecasting Vehicle Fleet Maintenance in Detroit
J. Gardner, D. Koutra, J. Mroueh, V. Pang, A. Farahi, S. Krassenstein, and J. Webb.
https://arxiv.org/abs/1710.06839
```

# Key files in this repo

## `tensor_preproc.py`

Python script to preprocess vehicles data into a format suitable for PARAFAC tensor decomposition in MATLAB. Writes output .tsv or .dat file for downstream analysis.

## `tensor_utils.py`

Utility functions used in `tensor_preproc.py`. This script is not executed by itself.

## `tensor_decomp.m`

MATLAB script which loads tensor toolbox and preprocessed data output from `tensor_preproc.py`, performs PARAFAC decomposition and plots the results over the time dimension. (Note: this script could be called from the command line or using the Python `subprocess` module at the end of tensor_preproc.py; we instead executed it manually using the Matlab GUI in order to ensure execution completed.)

# To extract data and build/evaluate LSTM:

First, create the files with maintenance sequences by make/model; each line represents a unique vehicle:
```$ python freq_pattern_preproc.py```

Second, clean up those sequences into a format the Tnesorflow code likes (this second preprocessing step should be built into the first step later):
```$ python lstm_preproc.py ```

Third, build the model (note that this script is in PYTHON 2; the other scripts are in PYTHON 3!):
```$ python2 ptb/ptb_word_lm.py  --data_path=/path/to/output/dir/from/lstm_preproc/ ```

```
2017-06-22 14:16:23.243476: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-22 14:16:23.243503: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-06-22 14:16:23.243508: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-22 14:16:23.243513: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
WARNING:tensorflow:Standard services need a 'logdir' passed to the SessionManager
Epoch: 1 Learning rate: 1.000
Epoch: 1 Train Perplexity: 593.959
Epoch: 1 Valid Perplexity: 114.834
Epoch: 2 Learning rate: 1.000
Epoch: 2 Train Perplexity: 77.896
Epoch: 2 Valid Perplexity: 41.869
Epoch: 3 Learning rate: 1.000
Epoch: 3 Train Perplexity: 48.058
Epoch: 3 Valid Perplexity: 47.353
Epoch: 4 Learning rate: 1.000
Epoch: 4 Train Perplexity: 38.648
Epoch: 4 Valid Perplexity: 29.654
Epoch: 5 Learning rate: 0.500
Epoch: 5 Train Perplexity: 23.264
Epoch: 5 Valid Perplexity: 22.252
Epoch: 6 Learning rate: 0.250
Epoch: 6 Train Perplexity: 17.788
Epoch: 6 Valid Perplexity: 15.532
Epoch: 7 Learning rate: 0.125
Epoch: 7 Train Perplexity: 16.802
Epoch: 7 Valid Perplexity: 15.344
Epoch: 8 Learning rate: 0.062
Epoch: 8 Train Perplexity: 16.635
Epoch: 8 Valid Perplexity: 15.284
...
```

Currently, code was written @jpgard (with MATLAB toolbox, examples, etc. from @dkoutra). LSTM code straight from the Google Tensorflow repository. Contact authors with questions/comments/etc.

