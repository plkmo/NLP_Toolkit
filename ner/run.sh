#!/bin/bash

MAIN_ROOT=$PWD/../..
SRC_ROOT=$MAIN_ROOT/src
export PATH=$SRC_ROOT/bin/:$PWD/utils/:$PATH
export PYTHONPATH=$SRC_ROOT/:$SRC_ROOT/data/:$SRC_ROOT/model/:$SRC_ROOT/solver/:$SRC_ROOT/utils/:$PYTHONPATH

python run_ner.py --data_dir data/ner/conll2003 --model_type bert --model_name_or_path bert-base-uncased --output_dir data/model_data --do_train --do_eval --evaluate_during_training --overwrite_output_dir
