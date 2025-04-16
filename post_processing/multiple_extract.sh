#!/bin/bash

# Check if extract.sh script exists
if [ ! -e extract.sh ]; then
    echo "Error: extras.sh script not found. Make sure it exists in the current directory."
    exit 1
fi

# Define file paths manually
file_paths=(
    #"/home/asrl/data2/2024-12-04-tunnel-fwd1/"
    #"/home/asrl/data2/2024-12-04-tunnel-fwd2/"
)

# Loop through the file paths and run extract.sh for each file
for file_path in "${file_paths[@]}"; do
    echo "Running extract.sh for file: $file_path"
    ./extract.sh "$file_path"
done

echo "Finished running extract.sh for all specified files."