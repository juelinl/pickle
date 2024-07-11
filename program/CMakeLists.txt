add_executable(benchmark benchmark.cpp dataloader.cpp
        ../include/adjlist.hpp)
target_link_libraries(benchmark)