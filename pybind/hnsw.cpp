//
// Created by juelin on 7/8/24.
//
#include "nsw_graph.hpp"
#include "nsw.hpp"
#include "timer.hpp"
#include "util.hpp"
#include "serializer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

namespace pickle {
#define ATEN_DTYPE_SWITCH(val, DType, ...)                                     \
  do {                                                                         \
    if ((val) == "uint8") {                                            \
      typedef uint8_t DType;                                                   \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == "int8") {                                      \
      typedef int8_t DType;                                                    \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == "float16") {                                   \
      typedef float16_t DType;                                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == "float32") {                                   \
      typedef float DType;                                                 \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      std::cerr << "DType can only be uint8, int8, float16, or float32\n";   \
      std::cerr << "Unsupported dtype: " << val;   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)


    struct HNSWGraph {
        std::vector<DynamicNSWGraphPtr> _graphs;
        size_t _search_ef{100};
        size_t _build_ef{100};
        size_t _top_k{10};
        size_t _M{32};
        DistanceFunction df{DistanceFunction::L2};
        py::array _data;

        HNSWGraph() = default;

        size_t getSearchEf() const { return _search_ef; };

        void setSearchEf(size_t ef) { _search_ef = ef; };

        size_t getBuildEf() const { return _build_ef; };

        void setBuildEf(size_t ef) { _build_ef = ef; };

        size_t getTopK() const { return _top_k; };

        void setTopK(size_t top_k) { _top_k = top_k; };

        size_t getM() const { return _M; };

        void setM(size_t M) { _M = M; };

        size_t getNumLayer() const { return _graphs.size(); };

        void build(py::array data) {
            _data = data;
            ALWAYS_ASSERT(!_data.is_none());
            ALWAYS_ASSERT(_data.ndim() == 2);
            size_t row = _data.shape(0);
            size_t dim = _data.shape(1);
            std::string dtype = py::str(_data.dtype());
            Timer timer;
            timer.start();
            std::vector<external_id_t> external_ids = getRandomIndices<external_id_t>(row);
            timer.end();
            std::cout << "Get random indices in " << timer.seconds() << "secs" << std::endl;

            timer.start();
            ATEN_DTYPE_SWITCH(dtype, T, {
                size_t count = _data.nbytes() / sizeof(T);
                ALWAYS_ASSERT(row * dim == count);
                std::span<const T> all_data = {static_cast<const T *>(_data.data()), count};
                _graphs = BuildNSWLayers<T>(df, _build_ef, _M, dim, external_ids, all_data);
                _graphs.at(0)->CreateMap();
            });
            timer.end();
            std::cout << "Build graph in " << timer.seconds() << "secs" << std::endl;
        }

        py::array_t <external_id_t> search(py::array query) {
            ALWAYS_ASSERT(!query.is_none());
            ALWAYS_ASSERT(query.shape(0) > 0);
            ALWAYS_ASSERT(query.shape(1) == _data.shape(1));
            ALWAYS_ASSERT(query.dtype().equal(_data.dtype()));
            ALWAYS_ASSERT(!_graphs.empty());

            size_t num_query = query.shape(0);
            size_t dim = query.shape(1);
            size_t entry_layer = _graphs.size() - 1;
            std::string dtype = py::str(query.dtype());
            py::array_t<external_id_t> result({num_query, _top_k});
            auto result_ptr = static_cast<external_id_t *>(result.request().ptr);

            #pragma omp parallel for schedule(static, 1)
            for (size_t i = 0; i < num_query; i++) {
                ATEN_DTYPE_SWITCH(dtype, T, {
                    size_t count = _data.nbytes() / sizeof(T);
                    std::span<const T> all_data = {static_cast<const T *>(_data.data()), count};
                    std::span<const T> q_data = {static_cast<const T *>(query.data(i)), dim};
                    auto ret = SearchNSWLayersSimple<T, std::dynamic_extent>(df, _top_k, _search_ef, dim, entry_layer,
                                                                             q_data, all_data, _graphs);
                    ALWAYS_ASSERT(ret.size() == _top_k);
                    for (size_t j = 0; j < ret.size(); j++){
                        result_ptr[i * _top_k + j] = _graphs.at(0)->GetExternalId(ret.at(j)._vid);
                    }
                });
            }
            return result;
        };

        void save(const std::string& filename) const {
            Serializer::to_disk(filename, _graphs);
        }

        void load(const std::string& filename) {
            _graphs = Serializer::from_disk(filename);
        }
    };
}

using namespace pickle;

PYBIND11_MODULE(pyann, m) {
    py::class_<HNSWGraph>(m, "HNSWGraph")
            .def(py::init<>())
            .def_property("search_ef", &HNSWGraph::getSearchEf, &HNSWGraph::setSearchEf)
            .def_property("build_ef", &HNSWGraph::getBuildEf, &HNSWGraph::setBuildEf)
            .def_property("top_k", &HNSWGraph::getTopK, &HNSWGraph::setTopK)
            .def_property("M", &HNSWGraph::getM, &HNSWGraph::setM)
            .def_property_readonly("num_layer", &HNSWGraph::getTopK)
            .def("save", &HNSWGraph::save, "Save the constructed index to the path")
            .def("load", &HNSWGraph::load, "Load the constructed index from the path")
            .def("build", &HNSWGraph::build, "Build hnsw graph on data with 2 dimension, supports int8, uint8, float16, and float32")
            .def("search", &HNSWGraph::search, "Search the query using the hnsw graph, supports int8, uint8, float16, and float32");
}