#!/bin/bash
work_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
url=https://knn-dataset.s3.us-east-2.amazonaws.com

echo "Download dataset into directory: $work_dir"

for graph_name in bigann deep msspacev text2image; do
    mkdir -p "$work_dir"/$graph_name
    pushd "$work_dir"/$graph_name || exit
    for data_size in "100K" "1M" "10M" "100M"; do
        wget $url/$graph_name/${graph_name}_data_${data_size}.npy
        wget $url/$graph_name/${graph_name}_label_${data_size}.npy
        wget $url/$graph_name/${graph_name}_distance_${data_size}.npy
    done
    wget $url/$graph_name/${graph_name}_query.npy
    popd || exit 
done