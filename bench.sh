#!/bin/bash

# shellcheck disable=SC2086

cur_dir="$(dirname "$(readlink -f "$0")")"
log_dir=${cur_dir}/log
data_dir=${cur_dir}/dataset/ann
index_dir=${cur_dir}/index
bin_dir=${cur_dir}/cmake-build-release/benchmark

# all_system=(official faiss pickle)
# all_data=(bigann deep msspacev text2image)
# all_size=("1M" "10M")
# all_build_ef=("100" "300" "500")
# max_degree=32


all_system=(pickle)
all_data=(deep)
all_size=("1M")
all_build_ef=("100")
max_degree=32

for data_name in "${all_data[@]}"; do

  mkdir -p $log_dir/"${data_name}"/

  space=l2
  if [[ "$data_name" == "text2image" ]]; then
    space=ip
  fi

  for build_ef in "${all_build_ef[@]}"; do
    for data_size in "${all_size[@]}"; do
      for system in "${all_system[@]}"; do

        bin_path=${bin_dir}/hnsw_${system}
        feat_path=${data_dir}/${data_name}/${data_name}_data_${data_size}.npy
        label_path=${data_dir}/${data_name}/${data_name}_label_${data_size}.npy
        distance_path=${data_dir}/${data_name}/${data_name}_distance_${data_size}.npy
        query_path=${data_dir}/${data_name}/${data_name}_query.npy
        index_path=${index_dir}/${data_name}_${data_size}_ef${build_ef}_M${max_degree}.hnsw_${system}
        log_path=${log_dir}/${data_name}/${data_name}_${data_size}_ef${build_ef}_M${max_degree}.hnsw_${system}

        runner=${cur_dir}/runner.sh
        $runner \
          --work_dir $cur_dir \
          --bin_path $bin_path \
          --space $space \
          --max_degree $max_degree \
          --build_ef $build_ef \
          --feat_path $feat_path \
          --label_path $label_path \
          --distance_path $distance_path \
          --query_path $query_path \
          --index_path $index_path \
          --log_path $log_path
      done
    done
  done
done
