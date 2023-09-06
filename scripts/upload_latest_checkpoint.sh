#!/bin/bash

# Check if experiment name is provided
if [ -z "$1" ]
then
    echo "No experiment name provided. Usage: ./scriptname.sh <experiment_name>"
    exit 1
fi

# Define the directory where the experiment results are
EXP_NAME=$1
EXP_DIR="../experiments/$EXP_NAME"

# Define the target s3 bucket
S3_BUCKET="s3://nmmo/model_weights"

# Find the latest checkpoint file
LATEST_FILE=$(find $EXP_DIR -name "*.pt" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -f2- -d" ")

# If a checkpoint file was found
if [ -n "$LATEST_FILE" ]; then
    # Extract filename
    FILENAME=$(basename -- "$LATEST_FILE")

    # Construct the target S3 path
    S3_PATH="$S3_BUCKET/$EXP_NAME.$FILENAME"

    # Copy the latest file to s3
    aws s3 cp $LATEST_FILE $S3_PATH
    echo "Copied $LATEST_FILE to $S3_PATH"
else
    echo "No checkpoint files found."
fi
