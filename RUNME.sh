#!/bin/bash
python LSTM_EMB.py  -c $@ -d $(date +%s) #-l "weights_8189_0_.hdf5"
#python traditional.py