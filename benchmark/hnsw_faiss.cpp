//
// Created by juelin on 7/26/24.
//

#include <iostream>
#include <faiss/IndexHNSW.h>
#include <memory>

#include "argparse.hpp"
#include "bench_util.hpp"

std::shared_ptr<faiss::IndexHNSW> build(Config config, Dataset dataset) {
    auto logger = GetLogger(config.log_path, "faiss_build");
    size_t num_row = dataset.feat.shape[0];
    size_t num_col = dataset.feat.shape[1];

    logger->info("Loading Data From: {}", config.feat_path);
    if (num_row >= 1000000) logger->info("Adding {}M points", num_row / 1000000);
    else logger->info("Adding {}K points", num_row / 1000);

    int d = num_col;
    int M = config.max_degree;

    faiss::MetricType metric = faiss::METRIC_L2;
    if (config.df == DistFunc::IP) {
        metric = faiss::METRIC_INNER_PRODUCT;
    } else if (config.df == DistFunc::L1) {
        metric = faiss::METRIC_L1;
    }

    size_t dataset_size = GetCurrentMemoryUsage();
    Timer timer;
    timer.start();
    auto index = std::make_shared<faiss::IndexHNSWFlat>(d, M, metric);
    index->add(num_row, dataset.feat.data<float>());
    timer.end();
    size_t graph_size = GetCurrentMemoryUsage() - dataset_size;
    logger->info("BuildTime={:.1f}s", timer.seconds());
    logger->info("NumThread={}", omp_get_max_threads());
    logger->info("IndexSize={}MB", graph_size);
    logger->info("DatasetSize={}MB", dataset_size);
    return index;
}

void bench(Config config, Dataset dataset, std::shared_ptr<faiss::IndexHNSW> index) {
    omp_set_num_threads(config.num_threads);
    ALWAYS_ASSERT(!config.query_path.empty());
    auto logger = GetLogger(config.log_path, "faiss_bench");
    size_t num_row = dataset.query.shape[0];
    size_t num_col = dataset.query.shape[1];

    std::vector<int> all_k{1, 10, 100};
    std::vector<int> all_search_ef{1, 5, 10, 20, 30, 50, 70, 90, 100, 200, 300};

    for (auto k: all_k) {
        for (auto search_ef: all_search_ef) {
            if (search_ef < k) continue;
            index->hnsw.efSearch = search_ef;
            index->hnsw.search_bounded_queue = false;
            faiss::hnsw_stats.reset();

            Timer timer;
            std::vector<faiss::idx_t> I(num_row * k);
            std::vector<float> D(num_row * k);

            timer.start();
            index->search(num_row, dataset.query.data<float>(), k, D.data(), I.data());
            timer.end();

            int total_matched = 0;
            for (int i = 0; i < num_row; i++) {
                auto labels = dataset.GetLabel(i);
                auto dists = dataset.GetDist(i);
                for (int m = 0; m < k; m++) {
                    auto p_dist = D.at(i * k + m);
                    auto p_label = I.at(i * k + m);
                    for (int j = 0; j < k; j++) {
                        // if (labels.at(j) == p_label || dists.at(j) >= p_dist) {
                        if (labels.at(j) == p_label) {
                            total_matched++;
                            break;
                        }
                    }
                }
            }

            double recall = 100.0 * total_matched / (k * num_row);
            double qps = 1.0 * num_row / timer.seconds() / config.num_threads;
            double hop = 1.0 * faiss::hnsw_stats.n2 / num_row;
            double dist = 1.0 * faiss::hnsw_stats.ndis / num_row;
            auto build_ef = config.build_ef;
            auto thread = omp_get_max_threads();
            logger->info("k={} search_ef={} recall={:.1f} qps/t={} hop={:.1f} dist={:.1f} build_ef={} thread={}", k,
                         search_ef, recall, int(qps), hop, dist, build_ef, thread);

        }
    }

}

int main(int argc, char *argv[]) {
    Config config(argc, argv);
    auto dataset = ToFloat(LoadDataset(config));
    auto index = build(config, dataset);
    bench(config, dataset, index);
}