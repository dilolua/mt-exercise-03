# MT Exercise 3: Pytorch RNN Language Models


# Clone the Repository:

Begin by cloning our repository from GitHub at https://github.com/dilolua/mt-exercise-03/tree/main.

# Update to download_data.sh:

We have updated the download_data.sh script to download "Bleak House" by Charles Dickens from https://www.gutenberg.org/cache/epub/1023/pg1023.txt.
This text will serve as the training dataset for our models.

# Update to train.sh:

- Path Resolution: The script now uses readlink -f to fully resolve all symbolic links and determine the base path accurately.
- Logging Directory: A new directory structure $scripts/task2/logs has been created for logging purposes.
- Device Specification: The device is set to "mps", enabling the use of Apple's Metal Performance Shaders (MPS) for GPU support.
- Dataset Update: The training dataset has been changed to --data $data/guthenberg.
- Dropout Settings: Multiple dropout values have been specified, including 0, 0.2, 0.5, 0.7, and 0.9, to explore various regularization strengths.
- Custom Logging: The --ppl-log parameter has been added for additional logging or performance tracking.

# Update to main.py:

.....

Modifications have been made to accommodate the updated training setup and data processing requirements.
After implementing these changes, you can proceed with training the models.

