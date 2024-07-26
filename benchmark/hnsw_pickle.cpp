//
// Created by juelin on 7/24/24.
//

#include <omp.h>
//#include "hnsw_v0.hpp"
#include "hnsw_v1.hpp"
#include "bench_util.hpp"

//using namespace pickle::v0;
using namespace pickle::v1;
std::shared_ptr<HNSWGraph> build(const Config& config, Dataset dataset) {
    auto logger = GetLogger(config.log_path, "pickle_build");
    size_t dataset_size = GetCurrentMemoryUsage();

    size_t num_row = dataset.feat.shape[0];
    size_t num_col = dataset.feat.shape[1];
    logger->info("Loading Data From: {}", config.feat_path);
    if (num_row >= 1000000) logger->info("Adding {}M points", num_row / 1000000);
    else logger->info("Adding {}K points", num_row / 1000);

    NDArray all_data(dataset.feat.data_holder, dataset.dtype, dataset.feat.shape);
    auto ext_ids = GetRandIndices<external_id_t>(num_row);

    Timer timer;
    timer.start();
    auto index = std::make_shared<HNSWGraph>(config.max_degree, config.build_ef, config.df);
    ATEN_DTYPE_SWITCH(dataset.dtype, DType, {
       index->Build<DType, std::dynamic_extent>(ext_ids, all_data);
    });
    timer.end();

    size_t graph_size = GetCurrentMemoryUsage() - dataset_size;

    logger->info("BuildTime={:.1f}s", timer.seconds());
    logger->info("IndexSize={}MB", graph_size);
    logger->info("DatasetSize={}MB", dataset_size);
    return index;
}

void bench(Config config, Dataset dataset, std::shared_ptr<HNSWGraph> index) {
    typedef std::vector<Entry> ResultType;

    auto logger = GetLogger(config.log_path, "pickle_bench");
//    logger->info("START BENCHMARK");
    size_t dataset_size = GetCurrentMemoryUsage();
    size_t num_row = dataset.query.shape[0];
    NDArray all_query(dataset.query.data_holder, dataset.dtype, dataset.query.shape);
    std::vector<int> all_k{1, 10, 100};
    std::vector<int> all_search_ef{1, 5, 10, 20, 30, 50, 70, 90, 100, 200, 300};

    for (auto k: all_k) {
        for (auto search_ef: all_search_ef) {
            if (search_ef < k) continue;

            Profiler::Global()->Reset();
            std::vector<ResultType> results;

            Timer timer;
            timer.start();
            ATEN_DTYPE_SWITCH(dataset.dtype, DType, {
                results = index->AnnSearch<DType, std::dynamic_extent>(k, search_ef, all_query);
            });
            timer.end();

            int total_matched = 0;
            for (int i = 0; i < num_row; i++) {
                auto search_result = results.at(i);
                auto labels = dataset.GetLabel(i);
                auto dists = dataset.GetDist(i);
                for (auto entry: search_result) {
                    auto p_dist = entry.m_dist;
                    auto p_label = entry.m_vid;
                    for (int j = 0; j < k; j++) {
                        if (labels.at(j) == p_label || dists.at(j) >= p_dist) {
                            total_matched++;
                            break;
                        }
                    }
                }
            }

            double recall = 100.0 * total_matched / (k * num_row);
            double qps = 1.0 * num_row / timer.seconds();
            double hop = 1.0 * Profiler::Global()->GetHop() / num_row;
            double dist = 1.0 * Profiler::Global()->GetDist() / num_row;
            double neighbor = 1.0 * Profiler::Global()->GetNeighbor() / num_row;
            auto build_ef = config.build_ef;
            logger->info("k={} search_ef={} recall={:.1f} qps={} hop={:.1f} neighbor={:.1f} dist={:.1f} build_ef={} ", k, search_ef, recall, int(qps), hop, neighbor, dist, build_ef);
        }
    }
//    logger->info("END BENCHMARK");
}

int main(int argc, char *argv[]) {
    Config config(argc, argv);
    auto dataset = LoadDataset(config);
    auto graph = build(config, dataset);
    bench(config, dataset, graph);
}