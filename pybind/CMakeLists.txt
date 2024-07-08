add_subdirectory(pybind11)

pybind11_add_module(pyann MODULE hnsw.cpp)
