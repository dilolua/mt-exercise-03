# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/dilolua/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh


# Update to download_data.sh:

We have updated the download_data.sh script to download "Bleak House" by Charles Dickens from https://www.gutenberg.org/cache/epub/1023/pg1023.txt.
This text will serve as the training dataset for our models.

# Update to train.sh:

- Path Resolution: The script now uses readlink -f to fully resolve all symbolic links and determine the base path accurately.
- Logging Directory: A new directory structure $scripts/task2/logs has been created for logging purposes.
- Device Specification: The device is set to "mps", enabling the use of Apple's Metal Performance Shaders (MPS) for GPU support.
- Dataset Update: The training dataset has been changed to --data $data/guthenberg.
- Dropout Settings: Multiple dropout values have been specified, including 0, 0.2, 0.5, 0.7, and 0.9, to explore various regularization strengths.
- Custom Logging: The --ppl-log parameter has been added for additional logging, performance tracking.

# Update to main.py (from /tools/pytorch-examples/word language model):

- Argument Parser: The --dropout argument in the updated version accepts multiple values (nargs="+"), allowing for different dropout settings to be tested simultaneously.
--ppl-log: A new argument --ppl-log was added to log the perplexities of each model after each epoch.

- Model Training: multiple models are initialized with varying dropout values (using a loop and the models_drpout dictionary) to test different configurations in parallel.
The evaluate function in the updated script takes an additional any_model parameter to specify which model to use for evaluation.
The model saving mechanism appends the dropout rate to the filename, so unique paths for models trained with different dropout rates can be saved.
If enabled by the --ppl-log argument, perplexities for different models and epochs are logged to training_log and validation_log.

- More about perplexity logging: Two dictionaries, training_log and validation_log, are initialized to store perplexity values. 
These dictionaries use dropout values as keys, and the values are lists that store the perplexity for each epoch.
After completing a batch of training, if the --ppl-log flag is enabled, the script calculates the perplexity from the current loss 
and appends this value to the corresponding list in training_log under the key for the current model’s dropout rate. 
This is done inside the train function right after printing the batch training statistics.
The script then calculates perplexity, where the loss is the average loss over the specified logging interval (args.log_interval).
After each epoch, the script evaluates the model using the validation dataset, calculates the validation loss, and converts this into perplexity. 
This perplexity is then logged to the validation_log dictionary, again using the model’s dropout rate as the key.
The validation perplexity for each epoch is appended to the list associated with the current dropout rate in the validation_log dictionary.

- Additional: included a function log_metrics_to_csv to save performance metrics like test perplexity into a CSV file for later analysis.

Modifications have been made to accommodate the updated training setup and data processing requirements.
After implementing these changes, you can proceed with training the model and generated text from a train model.




