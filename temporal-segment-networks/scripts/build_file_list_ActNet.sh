#!/usr/bin/env bash

DATASET=$1
FRAME_PATH=$2

python tools/build_file_list_ActNet.py ${DATASET} ${FRAME_PATH} --shuffle
