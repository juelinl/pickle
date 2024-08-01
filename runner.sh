#!/bin/bash

# shellcheck disable=SC2086

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --work_dir) work_dir="$2"; shift ;;
        --bin_path) bin_path="$2"; shift ;;
        --space) space="$2"; shift ;;
        --max_degree) max_degree="$2"; shift ;;
        --build_ef) build_ef="$2"; shift ;;
        --feat_path) feat_path="$2"; shift ;;
        --label_path) label_path="$2"; shift ;;
        --distance_path) distance_path="$2"; shift ;;
        --query_path) query_path="$2"; shift ;;
        --index_path) index_path="$2"; shift ;;
        --log_path) log_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd $work_dir || exit

$bin_path \
    --space $space \
    --max_degree $max_degree \
    --build_ef $build_ef \
    --feat_path $feat_path \
    --label_path $label_path \
    --distance_path $distance_path \
    --query_path $query_path \
    --index_path $index_path \
    --log_path $log_path