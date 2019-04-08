# item_embed_rec
Recommendation using item embedding based on user history
## GradientBoosting(machine learning using only CPU) version
gbc - using GradientBoosting for classification  
tf_wals - WALS(Weighted Alternating Least Sqaure by Tensorflow)  
wp_predict - method hub for many models

## din_model - base_model.py, din_model.py
Deep Interest Network by Alibaba, rewritten in tensorflow

## WALS
tf_wals.py - main for Weighted Alternating squares MF model
tf_wals_lib.py - wals model
tf_wals_low.py - low level version of wals model

## w2v_lr
using built-in linear classifier

## w2vgbc.py
using scikit-learn gradientboostingclassifier

## wp_rnn_37.py
bidirectional GRU

## preprocessing modules
many files,  make*,  preprocess*, *preprocess
