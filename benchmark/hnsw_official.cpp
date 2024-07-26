//
// Created by juelin on 7/23/24.
//
#include <hnswlib/hnswlib.h>
#include <omp.h>
#include "bench_util.hpp"

template<typename SpaceType, typename DistanceType>
void build(Config config, Dataset dataset) {
    typedef std::priority_queue<std::pair<DistanceType, hnswlib::labeltype>> ResultType;
    auto logger = GetLogger(config.log_path, "hnsw_build");

    size_t num_row = dataset.feat.shape[0];
    size_t num_col = dataset.feat.shape[1];

    logger->info("Loading Data From: {}", config.feat_path);
    if (num_row >= 1000000) logger->info("Adding {}M points", num_row / 1000000);
    else logger->info("Adding {}K points", num_row / 1000);

    auto ext_ids = GetRandIndices<size_t>(num_row);
    size_t dataset_size = GetCurrentMemoryUsage();
    SpaceType space(num_col);
    hnswlib::HierarchicalNSW<DistanceType> *alg_hnsw{nullptr};
    Timer timer;
    timer.start();
    ALWAYS_ASSERT(!config.feat_path.empty());
    alg_hnsw = new hnswlib::HierarchicalNSW<DistanceType>(
            &space, num_row, config.max_degree, config.build_ef);

#pragma omp parallel for schedule(static, 128)
    for (size_t i = 0; i < num_row; i++) {
        auto ext_id = ext_ids.at(i);
        alg_hnsw->addPoint(dataset.GetFeat<void>(ext_id), ext_id);
    };

    timer.end();
    size_t graph_size = GetCurrentMemoryUsage() - dataset_size;
    logger->info("BuildTime={:.1f}s", timer.seconds());
    logger->info("NumThread={}", omp_get_max_threads());
    logger->info("IndexSize={}MB", graph_size);
    logger->info("DatasetSize={}MB", dataset_size);
    alg_hnsw->saveIndex(config.index_path);
}

template<typename SpaceType, typename DistanceType>
void bench(Config config, Dataset dataset) {
    omp_set_num_threads(config.num_threads);
    typedef std::vector<std::pair<DistanceType, hnswlib::labeltype>> ResultType;
    ALWAYS_ASSERT(!config.query_path.empty());
    auto logger = GetLogger(config.log_path, "hnsw_bench");
    size_t num_row = dataset.query.shape[0];
    size_t num_col = dataset.query.shape[1];
    SpaceType space(num_col);

    auto alg_hnsw = new hnswlib::HierarchicalNSW<DistanceType>(&space, config.index_path);

    std::vector<int> all_k{1, 10, 100};
    std::vector<int> all_search_ef{1, 5, 10, 20, 30, 50, 70, 90, 100, 200, 300};

    for (auto k: all_k) {
        for (auto search_ef: all_search_ef) {
            if (search_ef < k) continue;
            std::vector<ResultType> results(num_row);
            alg_hnsw->setEf(search_ef);
            alg_hnsw->metric_distance_computations = 0;
            alg_hnsw->metric_hops = 0;
            Timer timer;
            timer.start();

#pragma omp parallel for schedule(dynamic, 20)
            for (size_t i = 0; i < num_row; i++) {
                results.at(i) = alg_hnsw->searchKnnCloserFirst(dataset.GetQuery<void>(i), k);
            };

            timer.end();
            int total_matched = 0;
            for (int i = 0; i < num_row; i++) {
                auto search_result = results.at(i);
                auto labels = dataset.GetLabel(i);
                auto dists = dataset.GetDist(i);
                for (auto [p_dist, p_label]: search_result) {
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
            double hop = 1.0 * alg_hnsw->metric_hops / num_row;
            double dist = 1.0 * alg_hnsw->metric_distance_computations / num_row;
            auto build_ef = config.build_ef;
            auto thread = omp_get_max_threads();
            logger->info("k={} search_ef={} recall={:.1f} qps/t={} hop={:.1f} dist={:.1f} build_ef={} thread={}", k,
                         search_ef, recall, int(qps), hop, dist, build_ef, thread);
        }
    }
}

int main(int argc, char *argv[]) {
    Config config(argc, argv);
    auto dataset = LoadDataset(config);
    omp_set_num_threads(config.num_threads);

    if (config.df == DistFunc::IP) {
        build<hnswlib::InnerProductSpace, float>(config, dataset);
        bench<hnswlib::InnerProductSpace, float>(config, dataset);
    } else if (config.df == DistFunc::L2) {
        if (dataset.dtype == DataType::Uint8) {
            build<hnswlib::L2SpaceI, int>(config, dataset);
            bench<hnswlib::L2SpaceI, int>(config, dataset);
        } else {
            dataset = ToFloat(dataset);
            build<hnswlib::L2Space, float>(config, dataset);
            bench<hnswlib::L2Space, float>(config, dataset);
        }
    } else {
        spdlog::error("Unsupported data format");
    }
}