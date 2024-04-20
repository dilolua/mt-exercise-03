#!/bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# Link default training data for easier access
mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# Download a different book - "Don Quixote" by Miguel de Cervantes
mkdir -p $data/bleak

mkdir -p $data/bleak/raw

wget https://www.gutenberg.org/cache/epub/1023/pg1023.txt
mv pg1023.txt $data/bleak/raw/bleak.txt

# Preprocess slightly
cat $data/bleak/raw/bleak.txt | python $base/scripts/preprocess_raw.py > $data/bleak/raw/bleak.cleaned.txt

# Tokenize, fix vocabulary upper bound
cat $data/bleak/raw/bleak.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/bleak/raw/bleak.preprocessed.txt

# Split into train, valid, and test sets
head -n 440 $data/bleak/raw/bleak.preprocessed.txt | tail -n 400 > $data/bleak/valid.txt
head -n 840 $data/bleak/raw/bleak.preprocessed.txt | tail -n 400 > $data/bleak/test.txt
tail -n 3075 $data/bleak/raw/bleak.preprocessed.txt | head -n 2955 > $data/bleak/train.txt

