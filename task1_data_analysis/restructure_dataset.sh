#!/bin/bash
set -e

DATA_DIR=/data

# Unzip the dataset
# This should ideally work well on a linux system, but it is extremely 
# slow on Windows. So, I recommend extracting the dataset beforehand.
# unzip $DATA_DIR/assignment_data_bdd.zip -d $DATA_DIR

# Create images folder
mkdir -p $DATA_DIR/images

# Move train and val images
mv $DATA_DIR/bdd100k_images_100k/bdd100k/images/100k/train $DATA_DIR/images/
mv $DATA_DIR/bdd100k_images_100k/bdd100k/images/100k/val $DATA_DIR/images/

# Remove original image folder
rm -rf $DATA_DIR/bdd100k_images_100k

# Move labels folder and rename
mv $DATA_DIR/bdd100k_labels_release/bdd100k/labels $DATA_DIR/labels
rm -rf $DATA_DIR/bdd100k_labels_release

echo "Dataset restructured! Proceeding with analysis..."
