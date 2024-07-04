add_executable(nsw_bench nsw_bench.cpp dataloader.cpp)
target_link_libraries(nsw_bench PRIVATE distance)