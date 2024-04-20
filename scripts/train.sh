#!/bin/bash

scripts=$(dirname "$0")
base=$(readlink -f $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models
mkdir -p $scripts/task2/logs

num_threads=4
device="mps"

echo "Starting training..."
SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
 python main.py --data $data/guthenberg \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0 0.2 0.5 0.7 0.9 --tied \
        --save $models/model.pt \
        --mps \
        --ppl-log)

echo "Training completed in $SECONDS seconds."
echo "Checking for CSV files..."

#mv test.csv $scripts/task_2/logs/test.csv
#mv validation.csv $scripts/task_2/logs/validation.csv
#mv training.csv $scripts/task_2/logs/training.csv