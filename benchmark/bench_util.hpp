//
// Created by juelin on 7/24/24.
//

#ifndef PICKLE_BENCH_UTIL_HPP
#define PICKLE_BENCH_UTIL_HPP

#include <thread>
#include <cnpy.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <queue>
#include <span>
#include <fstream>

#include "timer.hpp"
#include "argparse.hpp"
#include "common.hpp"
#include "util.hpp"

using namespace pickle;

static const std::vector<int> all_k = {1, 10, 100};
static const std::vector<int> all_search_ef = {1, 5, 10, 20, 30, 50, 70, 90, 100, 200, 300, 400, 500};

inline auto GetLogger(const std::string &filename, const std::string &name, bool truncate = true) {
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, truncate);
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    // Create a combined sink that duplicates messages to both file and stdout
    std::vector<spdlog::sink_ptr> sinks = {file_sink, stdout_sink};
    auto logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    spdlog::register_logger(logger);
    spdlog::set_level(spdlog::level::info);
//    spdlog::flush_on(spdlog::level::info);
    return logger;
};

// Function to get current memory usage (in MB)
inline unsigned long long GetCurrentMemoryUsage() {
    unsigned long long memoryUsage = 0;
    // Platform-specific code to get memory usage
    // For Linux
#ifdef __linux__
    std::ifstream stat_stream("/proc/self/statm");
    if (stat_stream.is_open()) {
        unsigned long long pages;
        stat_stream >> pages;  // Read memory usage in pages
        memoryUsage = pages * sysconf(_SC_PAGESIZE) / 1024 / 1024;  // Convert pages to MB
        stat_stream.close();
    }
#endif

    // For Windows
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            memoryUsage = pmc.PrivateUsage / 1024;  // Private memory usage in KB
        }
#endif

    return memoryUsage;
}

inline DataType GetNumpyType(std::string fname) {
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp) throw std::runtime_error("Unable to open file " + fname);
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    size_t loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    char type = header[loc1 + 1];
    std::string str_ws = header.substr(loc1 + 2);
    size_t loc2 = str_ws.find("'");
    auto word_size = atoi(str_ws.substr(0, loc2).c_str());

    fclose(fp);
    if (type == 'i' && word_size == 1) {
        return DataType::Int8;
    } else if (type == 'u' && word_size == 1) {
        return DataType::Uint8;
    } else if (type == 'f' && word_size == 2) {
        return DataType::Float16;
    } else if (type == 'f' && word_size == 4) {
        return DataType::Float32;
    }
    spdlog::error("Unsupported type {} and word size {} combination", type, word_size);
    exit(-1);
};

struct Dataset {
    cnpy::NpyArray label;
    cnpy::NpyArray distance;
    cnpy::NpyArray query;
    cnpy::NpyArray feat;
    DataType dtype; // feat and query data type

    template<class T>
    T *GetFeat(size_t idx) {
        return static_cast<T *>(feat.data<char>() + idx * feat.shape[1] * feat.word_size);
    };

    template<class T>
    T *GetQuery(size_t idx) {
        return static_cast<T *>(query.data<char>() + idx * query.shape[1] * query.word_size);
    };

    std::vector<int> GetLabel(size_t idx) {
        auto start = label.data<int>() + idx * label.shape[1];
        auto end = start + label.shape[1];
        return {start, end};
    }

    std::vector<float> GetDist(size_t idx) {
        auto start = distance.data<float>() + idx * distance.shape[1];
        auto end = start + distance.shape[1];
        return {start, end};
    }
};

inline cnpy::NpyArray ToFloat(DataType dtype, const cnpy::NpyArray &input) {
    if (dtype == pickle::DataType::Float32) {
        ALWAYS_ASSERT(input.word_size == sizeof(float));
        return input;
    }

    auto output = cnpy::NpyArray(input.shape, sizeof(float), false);
    ATEN_DTYPE_SWITCH(dtype, DType, {
        auto input_data = input.data<DType>();
        auto output_data = output.data<float>();
        auto total_elem = input.num_vals;
        for (size_t i = 0; i < input.num_vals; i++) {
            output_data[i] = static_cast<float>(input_data[i]);
        }
    });
    return output;
}

inline Dataset ToFloat(const Dataset &input) {
    if (input.dtype == DataType::Float32) {
        return input;
    }

    Dataset output = input;
    output.feat = ToFloat(input.dtype, input.feat);
    output.query = ToFloat(input.dtype, input.query);
    output.dtype = DataType::Float32;
    return output;
}

struct Config {
    DistFunc df{DistFunc::RUNTIME};
    size_t max_degree{32};
    size_t build_ef{100};
    size_t num_threads{1};

    std::string feat_path;
    std::string label_path;
    std::string query_path;
    std::string distance_path;
    std::string index_path;
    std::string log_path;

    Config() = default;

    Config(int argc, char *argv[]) { Init(argc, argv); };

    void Init(int argc, char *argv[]) {
        argparse::ArgumentParser program("HNSW build index binary");
        program.add_argument("--space").help("one of l2, ip, or cosine").required();

        program.add_argument("--max_degree")
                .help(" maximum number of outgoing connections in the graph")
                .scan<'u', size_t>()
                .default_value(32ul);

        program.add_argument("--build_ef")
                .help("priority queue capacity during the index construction")
                .scan<'u', size_t>()
                .default_value(100ul);

        size_t default_thread_num = std::thread::hardware_concurrency();
        program.add_argument("--num_threads")
                .help("max num threads")
                .scan<'u', size_t>()
                .default_value(default_thread_num);

        program.add_argument("--feat_path")
                .help("path to the feature file")
                .required();

        program.add_argument("--index_path")
                .help("path to the output index")
                .required();

        program.add_argument("--label_path")
                .help("path to the ground truth label file")
                .required();

        program.add_argument("--distance_path")
                .help("path to the ground truth distance file")
                .required();

        program.add_argument("--query_path")
                .help("path to the query file")
                .required();

        program.add_argument("--log_path")
                .help("path to the log file")
                .default_value("/tmp/out.log");

        program.parse_args(argc, argv);

        auto space = program.get<std::string>("--space");

        if (space == "l2") {
            df = pickle::DistFunc::L2;
        } else if (space == "ip") {
            df = pickle::DistFunc::IP;
        } else if (space == "l1") {
            df = pickle::DistFunc::L1;
        } else {
            std::cerr << "Unsupported space type: " << space << std::endl;
            exit(-1);
        }

        max_degree = program.get<size_t>("--max_degree");
        build_ef = program.get<size_t>("--build_ef");
        feat_path = program.get<std::string>("--feat_path");
        index_path = program.get<std::string>("--index_path");
        label_path = program.get<std::string>("--label_path");
        distance_path = program.get<std::string>("--distance_path");
        query_path = program.get<std::string>("--query_path");
        log_path = program.get<std::string>("--log_path");
        num_threads = program.get<size_t>("--num_threads");
    }
};

[[nodiscard]] inline Dataset LoadDataset(const Config &config) {
    ALWAYS_ASSERT(!config.distance_path.empty());
    ALWAYS_ASSERT(!config.label_path.empty());
    ALWAYS_ASSERT(!config.query_path.empty());
    ALWAYS_ASSERT(!config.feat_path.empty());

    Dataset dataset;
    dataset.distance = cnpy::npy_load(config.distance_path);
    dataset.label = cnpy::npy_load(config.label_path);
    dataset.query = cnpy::npy_load(config.query_path);
    dataset.feat = cnpy::npy_load(config.feat_path);
    dataset.dtype = GetNumpyType(config.feat_path);
    return dataset;
}

#endif //PICKLE_BENCH_UTIL_HPP
