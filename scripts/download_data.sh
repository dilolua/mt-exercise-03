#! /bin/bash

scripts=$(dirname "$0")
PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
base=$scripts/..

data=$base/data
tools=$base/tools

# link default training data for easier access
mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!
mkdir -p $data/guthenberg
mkdir -p $data/guthenberg/raw

wget https://www.gutenberg.org/cache/epub/1023/pg1023.txt -O $data/guthenberg/raw/bleak.txt

# Check if the download was successful before moving forward
if [ -f "$data/guthenberg/raw/bleak.txt" ]; then
    # preprocess slightly
    cat $data/guthenberg/raw/bleak.txt | python $base/scripts/preprocess_raw.py > $data/guthenberg/raw/bleak.cleaned.txt

    # tokenize, fix vocabulary upper bound
    cat $data/guthenberg/raw/bleak.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
        $data/guthenberg/raw/bleak.preprocessed.txt

    # split into train, valid, and test
    # Creating the validation set
    head -n 3105 $data/guthenberg/raw/bleak.preprocessed.txt | tail -n 3105 > $data/guthenberg/valid.txt
    # Creating the test set
    head -n 6210 $data/guthenberg/raw/bleak.preprocessed.txt | tail -n 3105 > $data/guthenberg/test.txt
    # Creating the training set
    tail -n 14590 $data/guthenberg/raw/bleak.preprocessed.txt > $data/guthenberg/train.txt

else
    echo "Error: Failed to download bleak.txt"
fi
