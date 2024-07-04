//
// Created by juelin on 6/30/24.
//

#include "argparse.hpp"
#include "dataloader.hpp"
#include "nsw.hpp"
#include "timer.hpp"
#include <iostream>

using namespace pickle;

struct Config {
    DistanceFunction df{DistanceFunction::L2};
    size_t M{0};
    size_t ef_construction{0};
    size_t max_elements{0};
    size_t num_threads{1};
    std::string feat_path;
    std::string index_path;
    std::string truth_path;
    std::string query_path;

    Config() = default;

    Config(int argc, char *argv[]) { Init(argc, argv); };

    void Init(int argc, char *argv[]) {
        size_t default_thread = 48;
        argparse::ArgumentParser program("HNSW build index binary");
        program.add_argument("--space").help("one of l2, ip, or cosine").required();
        program.add_argument("--M")
                .help(" maximum number of outgoing connections in the graph")
                .scan<'u', size_t>()
                .default_value(32ul);
        program.add_argument("--ef_construction")
                .help("priority queue capacity during the index construction")
                .scan<'u', size_t>()
                .default_value(200ul);
        program.add_argument("--max_elements")
                .help("max elements in the graph")
                .scan<'u', size_t>()
                .default_value(1000000ul); // 1M
        program.add_argument("--num_threads")
                .help("max num threads")
                .scan<'u', size_t>()
                .default_value(default_thread); // 1M
        program.add_argument("--feat_path")
                .help("path to the feature file")
                .required();
        program.add_argument("--index_path")
                .help("path to the output index")
                .required();
        program.add_argument("--truth_path")
                .help("path to the truth file")
                .required();
        program.add_argument("--query_path")
                .help("path to the query file")
                .required();
        program.parse_args(argc, argv);
        std::string space = program.get<std::string>("--space");
        if (space == "l2") {
            df = pickle::DistanceFunction::L2;
        } else if (space == "ip") {
            df = pickle::DistanceFunction::IP;
        } else if (space == "l1") {
            df = pickle::DistanceFunction::L1;
        } else {
            std::cerr << "Unsupported space type: " << space << std::endl;
            exit(-1);
        }
        M = program.get<size_t>("--M");
        ef_construction = program.get<size_t>("--ef_construction");
        max_elements = program.get<size_t>("--max_elements");
        feat_path = program.get<std::string>("--feat_path");
        index_path = program.get<std::string>("--index_path");
        truth_path = program.get<std::string>("--truth_path");
        query_path = program.get<std::string>("--query_path");
        num_threads = program.get<size_t>("--num_threads");
    }
};

inline int get_num_matched(int top_k, const GroundTruthPtr& truth, const std::vector<std::vector<Entry>>& results) {
    size_t num_queries = results.size();
    int total_matched{0};
    for (int i = 0; i < num_queries; i++) {
        auto search_result = results.at(i);
        for (auto res: search_result) {
            for (int j = 0; j < top_k; j++) {
                auto idx = i * truth->_shape[1] + j;
                auto t_label = truth->_label[idx];
                auto t_dist = truth->_distance[idx];
                if (t_label == res._vid || t_dist >= res._distance) {
                    total_matched++;
                    break;
                }
            }
        }
    }
    return total_matched;
};

int main(int argc, char *argv[]) {
    Config config(argc, argv);
    std::cout << "hello world!\n";

    Timer timer;
    timer.start();
    auto feat = LoadArray2D(config.feat_path, config.max_elements);
    auto dim = feat->_shape[1];
    timer.end();
    std::cout << "Load feature in " << timer.seconds() << "secs" << std::endl;

    timer.start();
    std::vector<external_id_t> external_ids =
            getRandomIndices<external_id_t>(config.max_elements);
    timer.end();
    std::cout << "Get random indices in " << timer.seconds() << "secs" << std::endl;

    timer.start();
    std::vector<DynamicNSWGraphPtr> graphs;
    ATEN_DTYPE_SWITCH(feat->_data_type, DType, {
//    auto graphs = BuildNSWLayer<DType>(config.df, config.ef_construction, config.M, dim,
//                          external_ids, feat->span<DType>());
        std::span<DType > all_data = feat->span<DType>();
        graphs = BuildNSWLayers<DType>(config.df, config.ef_construction, config.M, dim,
                                       external_ids, all_data);


    });
    timer.end();
    std::cout << "Finished in " << timer.seconds() << " secs" << std::endl;


    if (config.query_path != "" && config.truth_path != "") {
        auto query = LoadArray2D(config.query_path);
        auto truth = LoadGroundTruth(config.truth_path);
        ALWAYS_ASSERT(query->_shape[0] == truth->_shape[0]);
        ALWAYS_ASSERT(query->_shape[0] > 0);
        ALWAYS_ASSERT(query->_data_type == feat->_data_type);
        ALWAYS_ASSERT(!graphs.empty());

        size_t num_queries = query->_shape[0];
        size_t dim = query->_shape[1];
        size_t entry_layer = graphs.size() - 1;
        std::vector<size_t> ef_search_vec = {100, 200};
        std::vector<size_t> top_k_vec = {1, 10, 100};

        for (auto top_k : top_k_vec) {
            for (auto ef_search: ef_search_vec) {
                std::vector<std::vector<Entry>> results(num_queries);
                timer.start();
                ATEN_DTYPE_SWITCH(query->_data_type, DType, {
                    for (int i = 0; i < num_queries; i++) {
                        std::span<DType> q_data = query->get_span<DType>(i);
                        std::span<DType > all_data = feat->span<DType>();
                        results.at(i) = SearchNSWLayersSimple<DType, std::dynamic_extent>(config.df, top_k, ef_search, dim, entry_layer, q_data, all_data, graphs);
                    }
                });
                timer.end();
                int total_matched = get_num_matched(top_k, truth, results);
                double recall = 100.0 * total_matched / (top_k * num_queries);
                double duration = timer.seconds();
                int qps = num_queries / duration;
                std::cout << "top_k=" << top_k << " ef_search=" << ef_search << " recall=" << recall << " qps=" << qps << std::endl;
            }
        }


    }
}