#!/bin/bash

# define the root directory
root_dir="../experiments"

# define the S3 bucket
s3_bucket="s3://nmmo"

# iterate over all subdirectories
for dir in $(find $root_dir -type d); do

  # check if there are any .pt files in the directory
  if ls $dir/*.pt > /dev/null 2>&1; then

    # get the .pt files, sorted by name
    all_files=$(ls -v $dir/*.pt)
    files_to_keep=$(ls -v $dir/*.pt | tail -n 3)

    # iterate over all the files
    for file in $all_files; do

      # define the object path on S3
      s3_object_path="${s3_bucket}/$(basename $dir)/$(basename $file)"

      # check if the file is one of the files to keep
      if echo $files_to_keep | grep -q $file; then

        # check if the file exists on S3
        if ! aws s3 ls $s3_object_path > /dev/null 2>&1; then

          # upload the file to S3
          aws s3 cp $file $s3_object_path
          echo "Uploaded $file to $s3_object_path"

        else
          echo "File $s3_object_path already exists in S3, not uploading"

        fi

      else

        # remove the file
        rm $file
        echo "Deleted $file"

      fi
    done
  fi
done
